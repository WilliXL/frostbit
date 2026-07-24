//! Fold kernels. Each writes only into pre-sized arena slots (the arena's
//! `record` debug-asserts the no-runtime-allocation invariant) and dispatches
//! on the typed [`Data`] view, delegating the heavy lifting to [`super::simd`].

use crate::container::{as_bitmap_mut, Bitmap, Data, Run};
use crate::format::*;
use crate::ops::arena::{OpArena, SlotState};
use crate::ops::cursor::{ContainerRef, FoldScratch};
use crate::ops::plan::{plan_diff, plan_intersect, plan_trivial, plan_union, Op};
use crate::ops::source::Inputs;
use crate::ops::{run, simd};
use crate::{FrozenBitmap, FrozenBitmapView};

// --- intersection -----------------------------------------------------------

/// Tiny-input one-shot gate: below this, the capacity walk dominates the fold
/// (run cells carry ~18-byte payloads), so ∧/\ skip it via [`plan_trivial`].
/// The count cap bounds the trivial arena (`keys × B`).
const TRIVIAL_MAX_BYTES: usize = 16 << 10;
const TRIVIAL_MAX_KEYS: usize = 32;

fn trivial(views: &[FrozenBitmapView<'_>], drive: usize) -> bool {
    views[drive].num_containers() <= TRIVIAL_MAX_KEYS
        && views.iter().map(|v| v.as_bytes().len()).sum::<usize>() <= TRIVIAL_MAX_BYTES
}

/// N-way intersection (AND). Driven by the input with the fewest containers:
/// only its keys are visited, and the others are `advance_to`-skipped to each —
/// so a selective conjunct never forces a full walk of the large inputs.
pub fn intersect(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    if !views.is_empty() {
        let seed = (0..views.len()).min_by_key(|&i| views[i].num_containers()).unwrap();
        if trivial(views, seed) {
            let plan = plan_trivial(Op::Intersect, views, seed);
            let mut arena = OpArena::from_plan(&plan);
            crate::ops::plan::recycle(plan);
            intersect_fold(&mut arena, views);
            return arena.serialize();
        }
    }
    intersect_into(views).serialize()
}

/// AND, folded into a (pooled) arena left for the caller to fold further or
/// serialize — the tree evaluator chains these without a byte round-trip.
pub fn intersect_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_intersect(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::plan::recycle(plan);
    intersect_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (AND) into a pre-sized `arena`. The arena's slots come from the
/// manifest, so this does no sizing analysis — it only drives keys and folds.
pub fn intersect_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    // Statically disjoint key sets: the plan proved the result empty before
    // any input byte is read — skip the whole walk. (Trivial plans never hit
    // this: their slots are the driving input's keys.)
    if inputs.is_empty() || arena.num_slots() == 0 {
        return;
    }
    let seed = (0..inputs.len()).min_by_key(|&i| inputs.container_count(i)).unwrap();
    let mut driver = inputs.cursor(seed);
    let mut scratch = FoldScratch::take();
    let (others, refs) = scratch.borrow();
    for i in (0..inputs.len()).filter(|&i| i != seed) {
        others.push(inputs.cursor(i));
    }
    // Fold the most-selective (fewest-container) partners first: the per-key
    // presence check rejects sooner and the accumulator shrinks faster.
    others.sort_by_key(|c| c.container_count());

    while let Some(key) = driver.peek_key() {
        let seed_ref = driver.get();
        driver.advance();
        refs.clear();
        refs.push(seed_ref);
        let present = others.iter_mut().all(|c| {
            let hit = c.advance_to(key);
            if hit {
                refs.push(c.get());
            }
            hit
        });
        if present {
            let slot = arena.claim_key(key);
            intersect_key(arena, slot, key, refs);
        }
    }
}

fn intersect_key(arena: &mut OpArena, i: usize, key: u16, refs: &mut [ContainerRef<'_>]) {
    // Seed from the smallest-card container; its representation fixes the
    // accumulator (array or bitmap) for the whole fold, so it never outgrows
    // the slot. AND only ever shrinks it. One scan gathers the seed, the
    // native-run precondition, and the card spread together.
    let (mut seed, mut min_card, mut max_card) = (0usize, u32::MAX, 0u32);
    let (mut runs_ok, mut truns) = (true, 0usize);
    for (j, r) in refs.iter().enumerate() {
        if r.card < min_card {
            (seed, min_card) = (j, r.card);
        }
        max_card = max_card.max(r.card);
        if runs_ok && r.typ == CT_RUN {
            truns += r.num_runs();
        } else {
            runs_ok = false;
        }
    }
    // Wide, skewed folds: ascending-card partner order shrinks the accumulator
    // before the expensive partners arrive (each fold costs ~|acc|·log|p|).
    // Gated so homogeneous folds never pay the sort.
    if refs.len() >= 5 && max_card >= min_card.saturating_mul(2) {
        refs.sort_unstable_by_key(|r| r.card);
        seed = 0;
    }

    if refs[seed].card as usize <= ARRAY_MAX_SIZE {
        // Array accumulator. The first array×array fold merges the source
        // containers straight into the slot (one pass, no extraction); array
        // merges after that ping-pong slot↔scratch (no staging copy); run and
        // bitmap partners filter in place. At most one copy-back at the end.
        let mut acc = ArrayAcc::new();
        let mut first = true;
        for (j, p) in refs.iter().enumerate() {
            if j == seed {
                continue;
            }
            if std::mem::take(&mut first) {
                if let (Data::Array(sa), Data::Array(pa)) = (refs[seed].typed(), p.typed()) {
                    acc.card = simd::array_intersect(sa, pa, acc_u16(arena.slot_mut(i))) as u32;
                    continue;
                }
                acc.card = load_array(arena.slot_mut(i), refs[seed].typed());
            }
            if acc.card == 0 {
                break;
            }
            acc.fold(arena, i, p.typed(), true);
        }
        if first {
            // Single-input intersect: the seed is the result.
            acc.card = load_array(arena.slot_mut(i), refs[seed].typed());
        }
        let card = acc.finish(arena, i);
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
    } else if runs_ok && truns <= MAX_RUNS {
        // Dense run containers stay runs (O(runs), not O(bitmap)). Seed from
        // the min-card container so an annihilating fold empties soonest.
        refs.swap(0, seed);
        let (card, bytes) = run_fold(arena, i, &refs[0], &refs[1..], run::intersect);
        arena.record(key, CT_RUN, card, i, bytes);
    } else {
        // Bitmap accumulator: fold partners with a fused AND+count so the result
        // card is always known, and stop early once it empties — a high-fan-in
        // AND of dense inputs collapses to nothing after a few steps.
        load_bitmap(arena.slot_mut(i), refs[seed].typed());
        let mut card = refs[seed].card;
        for (j, p) in refs.iter().enumerate() {
            if j == seed {
                continue;
            }
            if card == 0 {
                break;
            }
            let (slot, scratch) = arena.slot_and_scratch(i);
            card = bitmap_and_count(acc(slot), scratch, p.typed());
        }
        let (typ, bytes) = finish_bitmap(arena, i, card);
        arena.record(key, typ, card, i, bytes);
    }
}

// --- union ------------------------------------------------------------------

/// N-way union (OR).
pub fn union(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    union_into(views).serialize()
}

/// OR, folded into a (pooled) arena for the caller to chain or serialize.
pub fn union_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_union(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::plan::recycle(plan);
    union_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (OR) into a pre-sized `arena` (no sizing analysis).
///
/// Partner-major: each pass streams ONE input leaf sequentially against the
/// accumulators resident in the arena; a slot seeds when a pass reaches its
/// key first. Array accumulators upgrade to bitmap form the moment their
/// running sum outgrows an array (only ever inside a bitmap-capped slot — the
/// plan proved array-capped keys stay small); run accumulators coalesce
/// natively and convert on overflow or a non-run partner. Deferred bitmap
/// counts fuse into the last pass that touches the slot.
pub fn union_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    if arena.is_double() {
        union_fold_passes(arena, inputs);
    } else {
        union_fold_keys(arena, inputs);
    }
}

/// Partner-major union (dispatched when the accumulator set is cache-resident;
/// see [`Plan::double`]).
fn union_fold_passes<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    for j in 0..inputs.len() {
        let mut c = inputs.cursor(j);
        let is_last = j + 1 == inputs.len();
        let mut s = 0usize;
        while let Some(key) = c.peek_key() {
            while arena.planned_key(s) < key {
                s += 1;
            }
            debug_assert_eq!(arena.planned_key(s), key, "input key absent from the plan");
            let p = c.get();
            c.advance();
            union_apply(arena, s, &p, is_last);
            s += 1;
        }
    }
    union_finalize(arena);
}

/// Seed slot `s` from its first container.
fn union_seed(arena: &mut OpArena, s: usize, p: &ContainerRef<'_>) {
    if arena.slot_capacity(s) < BITMAP_BYTES {
        // The plan proved this key's whole union stays array-sized.
        let card = load_array(arena.slot_mut(s), p.typed());
        set_state(arena, s, CT_ARRAY, 0, card);
        return;
    }
    match p.typ {
        CT_RUN => {
            let n = p.data.len();
            arena.slot_mut(s)[..n].copy_from_slice(p.data);
            set_state(arena, s, CT_RUN, p.num_runs() as u16, p.card);
        }
        _ => {
            load_bitmap(arena.slot_mut(s), p.typed());
            set_state(arena, s, CT_BITMAP, 0, p.card);
        }
    }
}

/// OR one partner container into slot `s`'s accumulator.
fn union_apply(arena: &mut OpArena, s: usize, p: &ContainerRef<'_>, is_last: bool) {
    let st = arena.state(s);
    if !st.seeded() {
        return union_seed(arena, s, p);
    }
    match st.typ {
        CT_ARRAY => {
            if (st.card + p.card) as usize > ARRAY_MAX_SIZE {
                // Unreachable by plan (array accs only seed in array-capped
                // slots, whose whole union stays array-sized) — but upgrade
                // rather than overflow if an input lies about its cardinality.
                debug_assert!(false, "array union acc outgrew an array-capped slot");
                array_acc_to_bitmap(arena, s);
                return union_apply(arena, s, p, is_last);
            }
            match p.typed() {
                Data::Array(b) => {
                    let (cur, other) = arena.slot_pair(s);
                    let src: &[u16] = &bytemuck::cast_slice(cur)[..st.card as usize];
                    let card = simd::array_union(src, b, bytemuck::cast_slice_mut(other)) as u32;
                    arena.flip_side(s);
                    arena.state_mut(s).card = card;
                }
                d => {
                    // Stage the partner as a sorted array, then merge.
                    let (cur, other, scratch) = arena.slot_pair_and_scratch(s);
                    let staged: &mut [u16] = bytemuck::cast_slice_mut(scratch);
                    let n = d.write_sorted(staged);
                    let src: &[u16] = &bytemuck::cast_slice(cur)[..st.card as usize];
                    let card =
                        simd::array_union(src, &staged[..n], bytemuck::cast_slice_mut(other)) as u32;
                    arena.flip_side(s);
                    arena.state_mut(s).card = card;
                }
            }
        }
        CT_RUN => {
            let bound = st.runs as usize + p.num_runs();
            if p.typ != CT_RUN || bound > MAX_RUNS {
                run_acc_to_bitmap(arena, s);
                return union_apply(arena, s, p, is_last);
            }
            let (cur, other) = arena.slot_pair(s);
            let src = slot_runs(cur, st.runs);
            let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut other[2..2 + bound * 4]);
            let (nr, card) = run::union(src, as_runs(p), dst);
            write_u16(other, 0, nr as u16);
            arena.flip_side(s);
            set_state(arena, s, CT_RUN, nr as u16, card);
        }
        _ => {
            let dst = acc(arena.slot_mut(s));
            if is_last {
                let card = or_into_count(dst, p.typed());
                arena.state_mut(s).card = card;
            } else {
                or_into(dst, p.typed());
                arena.state_mut(s).card = SlotState::CARD_LAZY;
            }
        }
    }
}

/// Record every seeded slot (resolving counts the last pass didn't fuse).
fn union_finalize(arena: &mut OpArena) {
    for s in 0..arena.num_slots() {
        let st = arena.state(s);
        if !st.seeded() {
            continue;
        }
        let key = arena.planned_key(s);
        match st.typ {
            CT_ARRAY => arena.record(key, CT_ARRAY, st.card, s, st.card as usize * 2),
            CT_RUN => arena.record(key, CT_RUN, st.card, s, 2 + st.runs as usize * 4),
            _ => {
                let card = if st.card == SlotState::CARD_LAZY {
                    simd::popcount(acc(arena.slot_mut(s)))
                } else {
                    st.card
                };
                arena.record(key, CT_BITMAP, card, s, BITMAP_BYTES);
            }
        }
    }
}

/// Scatter an array accumulator into bitmap form on the other side.
fn array_acc_to_bitmap(arena: &mut OpArena, s: usize) {
    let st = arena.state(s);
    let (cur, other) = arena.slot_pair(s);
    let dst = as_bitmap_mut(&mut other[..BITMAP_BYTES]);
    simd::clear(dst);
    simd::set_values(dst, &bytemuck::cast_slice(cur)[..st.card as usize]);
    arena.flip_side(s);
    set_state(arena, s, CT_BITMAP, 0, st.card);
}

/// Expand a run accumulator into bitmap form on the other side.
fn run_acc_to_bitmap(arena: &mut OpArena, s: usize) {
    let st = arena.state(s);
    let (cur, other) = arena.slot_pair(s);
    let dst = as_bitmap_mut(&mut other[..BITMAP_BYTES]);
    simd::clear(dst);
    simd::set_runs(dst, slot_runs(cur, st.runs));
    arena.flip_side(s);
    set_state(arena, s, CT_BITMAP, 0, st.card);
}

// --- difference -------------------------------------------------------------

/// N-way difference: `inputs[0]` minus the rest.
pub fn diff(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    if !views.is_empty() && trivial(views, 0) {
        let plan = plan_trivial(Op::Diff, views, 0);
        let mut arena = OpArena::from_plan(&plan);
        crate::ops::plan::recycle(plan);
        diff_fold(&mut arena, views);
        return arena.serialize();
    }
    diff_into(views).serialize()
}

/// DIFF, folded into a (pooled) arena for the caller to chain or serialize.
pub fn diff_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_diff(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::plan::recycle(plan);
    diff_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (DIFF: `inputs[0]` minus the rest) into a pre-sized `arena`.
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

#[inline]
fn set_state(arena: &mut OpArena, i: usize, typ: u8, runs: u16, card: u32) {
    let side = arena.state(i).side;
    *arena.state_mut(i) = SlotState { typ, side, runs, card };
}

/// Runs held in a slot, in wire layout (`u16` count + `(start, len)` pairs).
#[inline]
fn slot_runs(slot: &[u8], nr: u16) -> &[Run] {
    bytemuck::cast_slice(&slot[2..2 + nr as usize * 4])
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
            let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut other[2..2 + bound * 4]);
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
            CT_RUN => arena.record(key, CT_RUN, st.card, i, 2 + st.runs as usize * 4),
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


// --- key-major fold order (cache-heavy accumulator sets) ---------------------

/// Drive an op over its keys: at each key, gather the containers present
/// (advancing the cursors) and hand them to `per_key`. Slots are claimed by
/// `per_key`, so keys that produce nothing (e.g. AND misses) cost no slot.
fn fold_keys<I: Inputs + ?Sized>(
    inputs: &I,
    arena: &mut OpArena,
    mut per_key: impl FnMut(&mut OpArena, u16, &[ContainerRef<'_>]),
) {
    let mut scratch = FoldScratch::take();
    let (cursors, refs) = scratch.borrow();
    for i in 0..inputs.len() {
        cursors.push(inputs.cursor(i));
    }
    while let Some(key) = cursors.iter().filter_map(|c| c.peek_key()).min() {
        refs.clear();
        for c in cursors.iter_mut() {
            if c.peek_key() == Some(key) {
                refs.push(c.get());
                c.advance();
            }
        }
        per_key(arena, key, refs);
    }
}


/// Key-major union: gather each key's containers across all inputs and fold
/// them together — one hot accumulator per key. Dispatched when the
/// accumulator set outgrows the cache (dense keys), where revisiting every
/// slot per pass would thrash L1.
fn union_fold_keys<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    fold_keys(inputs, arena, |arena, key, refs| {
        let slot = arena.claim_key(key);
        union_key(arena, slot, key, refs);
    });
}


fn union_key(arena: &mut OpArena, i: usize, key: u16, refs: &[ContainerRef<'_>]) {
    // The plan already decided this key's accumulator form (its cap): a
    // bitmap-sized slot means bitmap (or native-run) accumulation. Deriving it
    // here from runtime refs could only disagree with the slot we were given.
    let total_runs: usize = refs.iter().map(|p| p.num_runs()).sum();
    let needs_bitmap = arena.slot_capacity(i) >= BITMAP_BYTES;

    if needs_bitmap && all_runs(refs) && total_runs <= MAX_RUNS {
        // Run ∪ Run stays a (coalesced) run container.
        let (card, bytes) = run_fold(arena, i, &refs[0], &refs[1..], run::union);
        arena.record(key, CT_RUN, card, i, bytes);
    } else if needs_bitmap {
        // Seed from a bitmap input (a single copy) rather than clearing then
        // OR-ing it back in, then OR the rest — fusing the count into the last.
        let seed = refs.iter().position(|p| p.typ == CT_BITMAP).unwrap_or(0);
        load_bitmap(arena.slot_mut(i), refs[seed].typed());
        let dst = acc(arena.slot_mut(i));
        let last = (0..refs.len()).rev().find(|&j| j != seed);
        let mut card = None;
        for (j, p) in refs.iter().enumerate() {
            if j == seed {
                continue;
            }
            if Some(j) == last {
                card = Some(or_into_count(dst, p.typed()));
            } else {
                or_into(dst, p.typed());
            }
        }
        let card = card.unwrap_or_else(|| simd::popcount(dst));
        arena.record(key, CT_BITMAP, card, i, BITMAP_BYTES);
    } else {
        // Array accumulator (see intersect_key: direct first fold + ping-pong).
        // Union has no in-place fold, so every partner is a merge; a non-array
        // partner is extracted into the second scratch half first.
        let mut acc = ArrayAcc::new();
        let mut first = true;
        for p in &refs[1..] {
            if std::mem::take(&mut first) {
                if let (Data::Array(a0), Data::Array(a1)) = (refs[0].typed(), p.typed()) {
                    acc.card = simd::array_union(a0, a1, acc_u16(arena.slot_mut(i))) as u32;
                    continue;
                }
                acc.card = load_array(arena.slot_mut(i), refs[0].typed());
            }
            acc.fold_union(arena, i, p.typed());
        }
        if first {
            // Single-input union: the input is the result.
            acc.card = load_array(arena.slot_mut(i), refs[0].typed());
        }
        let card = acc.finish(arena, i);
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
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
    slot[2..2 + nr * 4].copy_from_slice(&cur[..nr * 4]);
    (card, 2 + nr * 4)
}


// --- shared helpers ---------------------------------------------------------

/// Whether every container is run-encoded — the precondition for a native run
/// fold. Taken only inside a bitmap-sized slot, where a `≤ MAX_RUNS` run
/// container always fits.
#[inline]
fn all_runs(refs: &[ContainerRef<'_>]) -> bool {
    !refs.is_empty() && refs.iter().all(|r| r.typ == CT_RUN)
}

#[inline]
fn as_runs<'a>(r: &ContainerRef<'a>) -> &'a [Run] {
    match r.typed() {
        Data::Run(runs) => runs,
        _ => unreachable!("as_runs on a non-run container"),
    }
}

/// Fold `seed` then `partners` with a native run `op`, writing a `CT_RUN`
/// container into slot `i`. Scratch is split into two run buffers and the
/// result copied to the slot. Returns `(cardinality, data bytes)`.
fn run_fold(
    arena: &mut OpArena,
    i: usize,
    seed: &ContainerRef<'_>,
    partners: &[ContainerRef<'_>],
    op: impl Fn(&[Run], &[Run], &mut [Run]) -> (usize, u32),
) -> (u32, usize) {
    let (slot, scratch) = arena.slot_and_scratch(i);
    let (a, b) = scratch.split_at_mut(BITMAP_BYTES);

    // Fold 1 reads the seed's runs straight from its container; after that the
    // accumulator ping-pongs between the scratch halves (no per-fold copy).
    let (mut nr, mut card, mut in_b) = (seed.num_runs(), seed.card, false);
    let mut rest = partners.iter();
    match rest.next() {
        Some(p0) => {
            (nr, card) = op(as_runs(seed), as_runs(p0), bytemuck::cast_slice_mut(a));
        }
        None => {
            let s = as_runs(seed);
            bytemuck::cast_slice_mut::<u8, Run>(a)[..s.len()].copy_from_slice(s);
        }
    }
    for p in rest {
        if card == 0 {
            break;
        }
        let (src, dst): (&[u8], &mut [u8]) = if in_b { (b, a) } else { (a, b) };
        let src: &[Run] = &bytemuck::cast_slice(src)[..nr];
        (nr, card) = op(src, as_runs(p), bytemuck::cast_slice_mut(dst));
        in_b = !in_b;
    }

    let cur: &[u8] = if in_b { b } else { a };
    write_u16(slot, 0, nr as u16);
    slot[2..2 + nr * 4].copy_from_slice(&cur[..nr * 4]);
    (card, 2 + nr * 4)
}

/// Finish a bitmap accumulator: downgrade to an array when sparse enough
/// (cheaper downstream folds + smaller output), else keep the bitmap. The
/// boundary is half the array limit: above it, extraction (~0.5 ns/value)
/// buys at most a 2x-shrinking output while costing more than the fold that
/// produced it — a card-3900 extraction was 63% of a dense 4-way difference.
/// `serialize_compact` still canonicalizes terminal results. Returns the
/// recorded `(type, data bytes)`.
fn finish_bitmap(arena: &mut OpArena, i: usize, card: u32) -> (u8, usize) {
    if card == 0 || card > (ARRAY_MAX_SIZE / 2) as u32 {
        return (CT_BITMAP, BITMAP_BYTES);
    }
    let (slot, scratch) = arena.slot_and_scratch(i);
    let n = Data::new(CT_BITMAP, card, &slot[..BITMAP_BYTES])
        .write_sorted(bytemuck::cast_slice_mut(scratch));
    let bytes = n * 2;
    slot[..bytes].copy_from_slice(&scratch[..bytes]);
    (CT_ARRAY, bytes)
}

/// The first `BITMAP_BYTES` of a slot, as a mutable bitmap.
#[inline]
fn acc(slot: &mut [u8]) -> &mut Bitmap {
    as_bitmap_mut(&mut slot[..BITMAP_BYTES])
}

/// Write a container's lows into `slot` as a sorted `u16` array; returns count.
#[inline]
fn load_array(slot: &mut [u8], data: Data<'_>) -> u32 {
    data.write_sorted(bytemuck::cast_slice_mut(slot)) as u32
}

/// Populate `slot` as a bitmap from `data`. A bitmap source is copied in one
/// pass; others clear then set.
#[inline]
fn load_bitmap(slot: &mut [u8], data: Data<'_>) {
    let dst = acc(slot);
    if let Data::Bitmap(b) = data {
        simd::copy(dst, b);
    } else {
        simd::clear(dst);
        or_into(dst, data);
    }
}

/// `dst |= data`.
#[inline]
fn or_into(dst: &mut Bitmap, data: Data<'_>) {
    match data {
        Data::Array(a) => simd::set_values(dst, a),
        Data::Run(r) => simd::set_runs(dst, r),
        Data::Bitmap(b) => simd::or(dst, b),
        Data::Inline(ids) => ids.iter().for_each(|&v| set_bit(dst, v as u16)),
    }
}

/// `dst &= !data`.
#[inline]
fn clear_into(dst: &mut Bitmap, data: Data<'_>) {
    match data {
        Data::Array(a) => simd::clear_values(dst, a),
        Data::Run(r) => simd::clear_runs(dst, r),
        Data::Bitmap(b) => simd::andnot(dst, b),
        Data::Inline(ids) => ids.iter().for_each(|&v| clear_bit(dst, v as u16)),
    }
}

/// Bitmap accumulator `&= partner`, fused with a population count of the result
/// (one pass). Non-bitmap partners are materialized into `scratch` first.
#[inline]
fn bitmap_and_count(dst: &mut Bitmap, scratch: &mut [u8], data: Data<'_>) -> u32 {
    if let Data::Bitmap(b) = data {
        simd::and_count(dst, b)
    } else {
        let tmp = acc(scratch);
        simd::clear(tmp);
        or_into(tmp, data);
        simd::and_count(dst, tmp)
    }
}

/// `or_into` fused with a population count. A bitmap source counts in one pass;
/// others set then count.
#[inline]
fn or_into_count(dst: &mut Bitmap, data: Data<'_>) -> u32 {
    if let Data::Bitmap(b) = data {
        simd::or_count(dst, b)
    } else {
        or_into(dst, data);
        simd::popcount(dst)
    }
}

/// `clear_into` fused with a population count.
#[inline]
fn clear_into_count(dst: &mut Bitmap, data: Data<'_>) -> u32 {
    if let Data::Bitmap(b) = data {
        simd::andnot_count(dst, b)
    } else {
        clear_into(dst, data);
        simd::popcount(dst)
    }
}

/// The slot's bytes as a `u16` merge-output buffer.
#[inline]
fn acc_u16(slot: &mut [u8]) -> &mut [u16] {
    bytemuck::cast_slice_mut(slot)
}

/// Array accumulator for one key's fold. Array×array merges ping-pong between
/// the slot and the first scratch half (the merge kernels need `out` disjoint
/// from both inputs, so flipping sides replaces a per-fold staging copy); run
/// and bitmap partners filter in place. [`finish`](Self::finish) copies back at
/// most once, whatever the arity.
struct ArrayAcc {
    card: u32,
    in_scratch: bool,
}

impl ArrayAcc {
    fn new() -> Self {
        ArrayAcc { card: 0, in_scratch: false }
    }

    /// AND (`keep`) / DIFF (`!keep`) fold with one partner.
    fn fold(&mut self, arena: &mut OpArena, i: usize, partner: Data<'_>, keep: bool) {
        let (slot, scratch) = arena.slot_and_scratch(i);
        let (sa, sb) = scratch.split_at_mut(BITMAP_BYTES);
        match partner {
            Data::Array(b) => {
                let kernel: fn(&[u16], &[u16], &mut [u16]) -> usize =
                    if keep { simd::array_intersect } else { simd::array_diff };
                self.merge(slot, sa, b, kernel);
            }
            Data::Run(runs) => {
                let cur = if self.in_scratch { sa } else { slot };
                self.card = retain_runs(acc_u16(cur), self.card, runs, keep);
            }
            Data::Bitmap(b) => {
                let cur = if self.in_scratch { sa } else { slot };
                self.card = retain_bitmap(acc_u16(cur), self.card, b, keep);
            }
            _ => {
                // Inline partner: its lows are already sorted — stage them once
                // and take the SIMD merge instead of a probe per element.
                let staged: &mut [u16] = bytemuck::cast_slice_mut(sb);
                let n = partner.write_sorted(staged);
                let (src, dst): (&[u8], &mut [u8]) =
                    if self.in_scratch { (sa, slot) } else { (slot, sa) };
                let src: &[u16] = &bytemuck::cast_slice(src)[..self.card as usize];
                let kernel: fn(&[u16], &[u16], &mut [u16]) -> usize =
                    if keep { simd::array_intersect } else { simd::array_diff };
                self.card = kernel(src, &staged[..n], bytemuck::cast_slice_mut(dst)) as u32;
                self.in_scratch = !self.in_scratch;
            }
        }
    }

    /// OR fold with one partner (always a merge; a non-array partner is
    /// extracted into the second scratch half first).
    fn fold_union(&mut self, arena: &mut OpArena, i: usize, partner: Data<'_>) {
        let (slot, scratch) = arena.slot_and_scratch(i);
        let (sa, sb) = scratch.split_at_mut(BITMAP_BYTES);
        match partner {
            Data::Array(b) => self.merge(slot, sa, b, simd::array_union),
            _ => {
                let staged: &mut [u16] = bytemuck::cast_slice_mut(sb);
                let n = partner.write_sorted(staged);
                self.merge(slot, sa, &staged[..n], simd::array_union);
            }
        }
    }


    /// Merge the accumulator with `b` into the other buffer and flip sides.
    /// The output is bounded to its buffer, so a kernel's 8-lane block store
    /// falls back to its scalar tail at the edge instead of spilling.
    fn merge(
        &mut self,
        slot: &mut [u8],
        sa: &mut [u8],
        b: &[u16],
        kernel: fn(&[u16], &[u16], &mut [u16]) -> usize,
    ) {
        let (src, dst): (&[u8], &mut [u8]) =
            if self.in_scratch { (sa, slot) } else { (slot, sa) };
        let src: &[u16] = &bytemuck::cast_slice(src)[..self.card as usize];
        self.card = kernel(src, b, bytemuck::cast_slice_mut(dst)) as u32;
        self.in_scratch = !self.in_scratch;
    }

    /// Land the accumulator in the slot (one copy iff it ended in scratch).
    fn finish(self, arena: &mut OpArena, i: usize) -> u32 {
        if self.in_scratch && self.card > 0 {
            let n = self.card as usize * 2;
            let (slot, scratch) = arena.slot_and_scratch(i);
            slot[..n].copy_from_slice(&scratch[..n]);
        }
        self.card
    }
}

/// Galloping array ∩ / ∖ run: filter sorted `acc` by run membership in one
/// pass (O(card + runs)), keeping values inside runs when `keep_inside`.
#[inline]
fn retain_runs(acc: &mut [u16], card: u32, runs: &[Run], keep_inside: bool) -> u32 {
    let (mut ri, mut w) = (0usize, 0usize);
    for r in 0..card as usize {
        let v = acc[r];
        while ri < runs.len() && runs[ri].end() < v {
            ri += 1;
        }
        let inside = ri < runs.len() && runs[ri].start <= v;
        if inside == keep_inside {
            acc[w] = v;
            w += 1;
        }
    }
    w as u32
}

/// Keep `acc[..card]` values where `keep` holds, compacting in place.
#[inline]
/// Filter sorted `acc` by bitmap membership with the word array hoisted out of
/// the loop. Going through `Data::contains` re-matches the container enum on
/// every probe (~3ns each — the corpus audit's whole offender population);
/// this tight loop with branchless compaction runs at ~1ns.
fn retain_bitmap(acc: &mut [u16], card: u32, b: &Bitmap, keep_inside: bool) -> u32 {
    let mut w = 0usize;
    for r in 0..card as usize {
        let v = acc[r];
        let hit = (b[(v >> 6) as usize] >> (v & 63)) & 1 != 0;
        acc[w] = v;
        w += (hit == keep_inside) as usize;
    }
    w as u32
}


#[inline]
fn set_bit(dst: &mut Bitmap, lo: u16) {
    dst[lo as usize / 64] |= 1u64 << (lo as usize % 64);
}

#[inline]
fn clear_bit(dst: &mut Bitmap, lo: u16) {
    dst[lo as usize / 64] &= !(1u64 << (lo as usize % 64));
}
