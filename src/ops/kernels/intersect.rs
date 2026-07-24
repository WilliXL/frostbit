//! N-way intersection (AND).

use crate::container::Data;
use crate::format::*;
use crate::ops::arena::OpArena;
use crate::ops::cursor::{ContainerRef, FoldScratch};
use crate::ops::analyze::plan::{plan_intersect, plan_trivial, Op};
use crate::ops::source::Inputs;
use crate::ops::run;
use crate::simd;
use crate::{FrozenBitmap, FrozenBitmapView};
use super::common::*;

// --- intersection -----------------------------------------------------------


/// N-way intersection (AND), in op-ready standard form. Driven by the input with
/// the fewest containers: only its keys are visited, and the others are
/// `advance_to`-skipped to each — so a selective conjunct never forces a full
/// walk of the large inputs. Degenerate inputs: `&[]` returns the empty set; a
/// single input is copied.
pub fn intersect(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    intersect_arena(views).serialize()
}

/// Like [`intersect`], but serialized to the smallest (compact) form for
/// storage rather than op-ready form.
pub fn intersect_compact(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    intersect_arena(views).serialize_compact()
}

/// Fold an AND into a pooled arena (shared by [`intersect`] / [`intersect_compact`]).
fn intersect_arena(views: &[FrozenBitmapView<'_>]) -> OpArena {
    if !views.is_empty() {
        let seed = (0..views.len()).min_by_key(|&i| views[i].num_containers()).unwrap();
        if trivial(views, seed) {
            let plan = plan_trivial(Op::Intersect, views, seed);
            let mut arena = OpArena::from_plan(&plan);
            crate::ops::analyze::plan::recycle(plan);
            intersect_fold(&mut arena, views);
            return arena;
        }
    }
    intersect_into(views)
}

/// AND, folded into a (pooled) arena left for the caller to fold further or
/// serialize — the tree evaluator chains these without a byte round-trip.
pub fn intersect_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let plan = plan_intersect(inputs);
    let mut arena = OpArena::from_plan(&plan);
    crate::ops::analyze::plan::recycle(plan);
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

