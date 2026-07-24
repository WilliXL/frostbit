//! N-way difference (`inputs[0]` minus the rest).

use crate::container::{Data, Run};
use crate::format::*;
use crate::ops::arena::{OpArena, SlotState};
use crate::ops::cursor::{ContainerRef, FoldScratch};
use crate::ops::analyze::plan::{plan_diff, plan_trivial, Op};
use crate::ops::source::Inputs;
use crate::ops::run;
use crate::simd;
use crate::{FrozenBitmap, FrozenBitmapView};
use super::common::*;

// --- difference -------------------------------------------------------------

/// N-way difference (`inputs[0]` minus the rest), in op-ready standard form.
/// Degenerate inputs: `&[]` returns the empty set; a single input is copied.
pub fn diff(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    diff_arena(views).serialize()
}

/// Like [`diff`], but serialized to the smallest (compact) form for storage.
pub fn diff_compact(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    diff_arena(views).serialize_compact()
}

/// Fold a DIFF into a pooled arena (shared by [`diff`] / [`diff_compact`]).
fn diff_arena(views: &[FrozenBitmapView<'_>]) -> OpArena {
    if !views.is_empty() && trivial(views, 0) {
        let plan = plan_trivial(Op::Diff, views, 0);
        let mut arena = OpArena::from_plan(&plan);
        crate::ops::analyze::plan::recycle(plan);
        diff_fold(&mut arena, views);
        return arena;
    }
    diff_into(views)
}

/// DIFF, folded into a (pooled) arena for the caller to chain or serialize.
pub fn diff_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_diff(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::analyze::plan::recycle(plan);
    diff_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (DIFF: `inputs[0]` minus the rest) into a pre-sized `arena`.
///
/// Partner-major: pass 1 walks the lhs fused with the first subtrahend
/// (array×array folds straight from the sources into the slot; bitmap−bitmap
/// is a fused counted andnot); each later pass streams ONE subtrahend leaf
/// sequentially against the accumulators resident in the arena; finalize
/// resolves deferred bitmap counts and records. Two linear streams per pass
/// instead of one interrupted stream per partner per key — the key-major
/// order was memory-bound at high fan-in (see benchmarks/report.py profile).
pub fn diff_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    if arena.is_double() {
        diff_fold_passes(arena, inputs);
    } else {
        diff_fold_keys(arena, inputs);
    }
}

/// Partner-major difference (dispatched when the accumulator set is
/// cache-resident; see [`Plan::double`]).
fn diff_fold_passes<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    if inputs.is_empty() {
        return;
    }
    let last = inputs.len() - 1;

    // Pass 1: seed every lhs key, fused with the first subtrahend.
    let mut lhs = inputs.cursor(0);
    let mut p1 = if inputs.len() > 1 { Some(inputs.cursor(1)) } else { None };
    while let Some(key) = lhs.peek_key() {
        let l = lhs.get();
        lhs.advance();
        let i = arena.claim_key(key);
        let first = p1.as_mut().and_then(|c| c.advance_to(key).then(|| c.get()));
        diff_seed(arena, i, &l, first.as_ref(), last == 1);
    }

    // Passes 2..n: one subtrahend, streamed against resident accumulators.
    for j in 2..inputs.len() {
        let mut c = inputs.cursor(j);
        for i in 0..arena.num_slots() {
            let st = arena.state(i);
            if !st.seeded() || st.card == 0 {
                continue;
            }
            if c.peek_key().is_none() {
                break;
            }
            if !c.advance_to(arena.planned_key(i)) {
                continue;
            }
            let p = c.get();
            diff_apply(arena, i, &p, j == last);
        }
    }

    diff_finalize(arena);
}

/// Seed slot `i` with the lhs container — verbatim, so a key no subtrahend
/// ever touches keeps its (tightly-capped) stored form — fusing the first
/// subtrahend's fold when it shares the key.
fn diff_seed(
    arena: &mut OpArena,
    i: usize,
    l: &ContainerRef<'_>,
    p: Option<&ContainerRef<'_>>,
    is_last: bool,
) {
    // Fused fast paths: fold the first subtrahend straight from the sources.
    if let Some(pr) = p {
        match (l.typed(), pr.typed()) {
            (Data::Array(la), Data::Array(ra)) => {
                let card = simd::array_diff(la, ra, acc_u16(arena.slot_mut(i))) as u32;
                set_state(arena, i, CT_ARRAY, 0, card);
                return;
            }
            (Data::Bitmap(lb), Data::Bitmap(rb)) if l.card as usize > ARRAY_MAX_SIZE => {
                // One-pass `lhs & !rhs`, counted — no load_bitmap copy.
                let card = simd::andnot_into_count(acc(arena.slot_mut(i)), lb, rb);
                set_state(arena, i, CT_BITMAP, 0, card);
                return;
            }
            _ => {}
        }
    }

    // A small container seeds as an array whenever its slot can hold one —
    // true for every key the plan saw a subtrahend touch (shrink-capped), so a
    // later pass always finds a foldable accumulator. Everything else seeds
    // verbatim: an untouched key keeps its stored form at its tight cap, and a
    // dense run/bitmap keeps its cheap fold form in a bitmap-sized slot.
    let small = l.card as usize <= ARRAY_MAX_SIZE;
    if small && (l.typ == CT_ARRAY || l.typ == CT_INLINE || arena.slot_capacity(i) >= l.card as usize * 2)
    {
        let card = load_array(arena.slot_mut(i), l.typed());
        set_state(arena, i, CT_ARRAY, 0, card);
    } else {
        match l.typ {
            CT_RUN => {
                let n = l.data.len();
                arena.slot_mut(i)[..n].copy_from_slice(l.data);
                set_state(arena, i, CT_RUN, l.num_runs() as u16, l.card);
            }
            _ => {
                load_bitmap(arena.slot_mut(i), l.typed());
                set_state(arena, i, CT_BITMAP, 0, l.card);
            }
        }
    }
    if let Some(pr) = p {
        diff_apply(arena, i, pr, is_last);
    }
}

/// Subtract one partner container from slot `i`'s accumulator.
fn diff_apply(arena: &mut OpArena, i: usize, p: &ContainerRef<'_>, is_last: bool) {
    let st = arena.state(i);
    // Non-array accumulators only arise from dense (card > 4096) seeds, whose
    // slots are bitmap-sized — every form below fits.
    debug_assert!(
        st.typ == CT_ARRAY || arena.slot_capacity(i) >= BITMAP_BYTES,
        "dense accumulator in an undersized slot"
    );

    match st.typ {
        CT_ARRAY => match p.typed() {
            Data::Array(b) => {
                let (cur, other) = arena.slot_pair(i);
                let src: &[u16] = &bytemuck::cast_slice(cur)[..st.card as usize];
                let card = simd::array_diff(src, b, bytemuck::cast_slice_mut(other)) as u32;
                arena.flip_side(i);
                arena.state_mut(i).card = card;
            }
            Data::Run(runs) => {
                let card = retain_runs(acc_u16(arena.slot_mut(i)), st.card, runs, false);
                arena.state_mut(i).card = card;
            }
            Data::Bitmap(b) => {
                let card = retain_bitmap(acc_u16(arena.slot_mut(i)), st.card, b, false);
                arena.state_mut(i).card = card;
            }
            d => {
                // Inline partner: stage its sorted lows and SIMD-merge.
                let (cur, other, scratch) = arena.slot_pair_and_scratch(i);
                let staged: &mut [u16] = bytemuck::cast_slice_mut(scratch);
                let n = d.write_sorted(staged);
                let src: &[u16] = &bytemuck::cast_slice(cur)[..st.card as usize];
                let card = simd::array_diff(src, &staged[..n], bytemuck::cast_slice_mut(other)) as u32;
                arena.flip_side(i);
                arena.state_mut(i).card = card;
            }
        },
        CT_RUN => {
            // Splitting can add a fragment per subtrahend item; leave run form
            // before the count can overflow. (A run accumulator here means
            // card > 4096, so the slot is bitmap-sized — either form fits.) A
            // bitmap/inline partner can't be subtracted in run form at all, so
            // it forces a bitmap conversion; branch on the partner type
            // explicitly (as `union_apply` does) rather than via an arithmetic
            // sentinel, which would overflow `st.runs + usize::MAX`.
            let (splittable, extra) = match p.typed() {
                Data::Run(r) => (true, r.len()),
                Data::Array(a) => (true, a.len()),
                _ => (false, 0),
            };
            if !splittable || st.runs as usize + extra > MAX_RUNS {
                run_acc_to_bitmap(arena, i);
                return diff_apply(arena, i, p, is_last);
            }
            let bound = st.runs as usize + extra;
            let (cur, other) = arena.slot_pair(i);
            let src = slot_runs(cur, st.runs);
            let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut other[2..run_bytes(bound)]);
            let (nr, card) = match p.typed() {
                Data::Run(rr) => run::diff(src, rr, dst),
                Data::Array(ra) => run::diff_array(src, ra, dst),
                _ => unreachable!("a non-splittable partner converts to bitmap above"),
            };
            write_u16(other, 0, nr as u16);
            arena.flip_side(i);
            set_state(arena, i, CT_RUN, nr as u16, card);
        }
        _ => {
            let dst = acc(arena.slot_mut(i));
            if is_last {
                let card = clear_into_count(dst, p.typed());
                arena.state_mut(i).card = card;
            } else {
                clear_into(dst, p.typed());
                arena.state_mut(i).card = SlotState::CARD_LAZY;
            }
        }
    }
}

/// Resolve deferred counts, demote sparse bitmaps, and record every survivor.
fn diff_finalize(arena: &mut OpArena) {
    for i in 0..arena.num_slots() {
        let st = arena.state(i);
        if !st.seeded() || st.card == 0 {
            continue;
        }
        let key = arena.planned_key(i);
        match st.typ {
            CT_ARRAY => arena.record(key, CT_ARRAY, st.card, i, st.card as usize * 2),
            CT_RUN => arena.record(key, CT_RUN, st.card, i, run_bytes(st.runs as usize)),
            _ => {
                let card = if st.card == SlotState::CARD_LAZY {
                    simd::popcount(acc(arena.slot_mut(i)))
                } else {
                    st.card
                };
                let (typ, bytes) = finish_bitmap(arena, i, card);
                arena.record(key, typ, card, i, bytes);
            }
        }
    }
}


/// Key-major difference (see `union_fold_keys` for when this order wins).
fn diff_fold_keys<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    if inputs.is_empty() {
        return;
    }
    let mut a = inputs.cursor(0);
    let mut scratch = FoldScratch::take();
    let (rhs, refs) = scratch.borrow();
    for i in 1..inputs.len() {
        rhs.push(inputs.cursor(i));
    }
    while let Some(key) = a.peek_key() {
        let lhs = a.get();
        a.advance();
        refs.clear();
        for c in rhs.iter_mut() {
            if c.advance_to(key) {
                refs.push(c.get());
            }
        }
        let slot = arena.claim_key(key);
        diff_key(arena, slot, key, &lhs, refs);
    }
}


fn diff_key(arena: &mut OpArena, i: usize, key: u16, lhs: &ContainerRef<'_>, rhs: &[ContainerRef<'_>]) {
    if rhs.is_empty() {
        // No subtrahend at this key: copy LHS verbatim (inline → array form).
        if lhs.typ == CT_INLINE {
            let card = load_array(arena.slot_mut(i), lhs.typed());
            arena.record(key, CT_ARRAY, card, i, card as usize * 2);
        } else {
            let n = lhs.data.len();
            arena.slot_mut(i)[..n].copy_from_slice(lhs.data);
            arena.record(key, lhs.typ, lhs.card, i, n);
        }
        return;
    }

    if lhs.card as usize <= ARRAY_MAX_SIZE {
        // Array accumulator (see intersect_key: direct first fold + ping-pong).
        let mut acc = ArrayAcc::new();
        let mut first = true;
        for p in rhs {
            if std::mem::take(&mut first) {
                if let (Data::Array(la), Data::Array(ra)) = (lhs.typed(), p.typed()) {
                    acc.card = simd::array_diff(la, ra, acc_u16(arena.slot_mut(i))) as u32;
                    continue;
                }
                acc.card = load_array(arena.slot_mut(i), lhs.typed());
            }
            if acc.card == 0 {
                break;
            }
            acc.fold(arena, i, p.typed(), false);
        }
        let card = acc.finish(arena, i);
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
    } else if lhs.typ == CT_RUN && diff_run_bound(lhs, rhs).is_some_and(|n| n <= MAX_RUNS) {
        // Dense run minus runs/arrays stays a run container (split, no expand).
        let (card, bytes) = run_fold_diff(arena, i, lhs, rhs);
        arena.record(key, CT_RUN, card, i, bytes);
    } else {
        // One dense bitmap minus one bitmap: fuse `slot = lhs & !rhs` in a single
        // pass, skipping the `load_bitmap` copy of lhs.
        if let [r0] = rhs {
            if let (Data::Bitmap(lb), Data::Bitmap(rb)) = (lhs.typed(), r0.typed()) {
                let card = simd::andnot_into_count(acc(arena.slot_mut(i)), lb, rb);
                let (typ, bytes) = finish_bitmap(arena, i, card);
                arena.record(key, typ, card, i, bytes);
                return;
            }
        }
        load_bitmap(arena.slot_mut(i), lhs.typed());
        let dst = acc(arena.slot_mut(i));
        let mut card = 0;
        for (idx, p) in rhs.iter().enumerate() {
            if idx + 1 == rhs.len() {
                card = clear_into_count(dst, p.typed());
            } else {
                clear_into(dst, p.typed());
            }
        }
        let (typ, bytes) = finish_bitmap(arena, i, card);
        arena.record(key, typ, card, i, bytes);
    }
}


/// Worst-case output run count for `lhs \ rhs` keeping `lhs` in run form — or
/// `None` if a subtrahend is a bitmap/inline (forces a bitmap result). Each
/// array point and each run boundary can add one fragment.
fn diff_run_bound(lhs: &ContainerRef<'_>, rhs: &[ContainerRef<'_>]) -> Option<usize> {
    let mut total = lhs.num_runs();
    for r in rhs {
        match r.typ {
            CT_RUN => total += r.num_runs(),
            CT_ARRAY => total += r.card as usize,
            _ => return None,
        }
    }
    Some(total)
}


/// `lhs \ rhs` keeping runs: fragment `lhs` runs around run/array subtrahends
/// (no bitmap expansion). Writes a `CT_RUN` container to slot `i`.
fn run_fold_diff(
    arena: &mut OpArena,
    i: usize,
    lhs: &ContainerRef<'_>,
    rhs: &[ContainerRef<'_>],
) -> (u32, usize) {
    let (slot, scratch) = arena.slot_and_scratch(i);
    let (a, b) = scratch.split_at_mut(BITMAP_BYTES);

    // Fold 1 reads the lhs runs straight from its container; after that the
    // accumulator ping-pongs between the scratch halves (see run_fold).
    let step = |src: &[Run], p: &ContainerRef<'_>, dst: &mut [u8]| match p.typed() {
        Data::Run(pr) => run::diff(src, pr, bytemuck::cast_slice_mut(dst)),
        Data::Array(pa) => run::diff_array(src, pa, bytemuck::cast_slice_mut(dst)),
        _ => unreachable!("run_fold_diff partner must be run or array"),
    };
    let (mut nr, mut card, mut in_b) = (lhs.num_runs(), lhs.card, false);
    let mut rest = rhs.iter();
    if let Some(p0) = rest.next() {
        (nr, card) = step(as_runs(lhs), p0, a);
    }
    for p in rest {
        if card == 0 {
            break;
        }
        let (src, dst): (&[u8], &mut [u8]) = if in_b { (b, a) } else { (a, b) };
        let src: &[Run] = &bytemuck::cast_slice(src)[..nr];
        (nr, card) = step(src, p, dst);
        in_b = !in_b;
    }

    let cur: &[u8] = if in_b { b } else { a };
    write_u16(slot, 0, nr as u16);
    slot[2..run_bytes(nr)].copy_from_slice(&cur[..nr * 4]);
    (card, run_bytes(nr))
}


