//! N-way union (OR).

use crate::container::{Data, Run};
use crate::format::*;
use crate::ops::arena::{OpArena, SlotState};
use crate::ops::cursor::ContainerRef;
use crate::ops::analyze::plan::plan_union;
use crate::ops::source::Inputs;
use crate::ops::kernels::run;
use crate::ops::simd;
use crate::{FrozenBitmap, FrozenBitmapView};
use super::accum::*;

// --- union ------------------------------------------------------------------

/// N-way union (OR), in op-ready standard form.
/// Degenerate inputs: `&[]` returns the empty set; a single input is copied.
pub fn union(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    union_into(views).serialize()
}

/// Like [`union`], but serialized to the smallest (compact) form for storage.
pub fn union_compact(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    union_into(views).serialize_compact()
}

/// OR, folded into a (pooled) arena for the caller to chain or serialize.
pub fn union_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_union(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::analyze::plan::recycle(plan);
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
            let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut other[2..run_bytes(bound)]);
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
            CT_RUN => arena.record(key, CT_RUN, st.card, s, run_bytes(st.runs as usize)),
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
