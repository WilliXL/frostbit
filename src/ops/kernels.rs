//! Fold kernels. Each writes only into pre-sized arena slots (the arena's
//! `record` debug-asserts the no-runtime-allocation invariant) and dispatches
//! on the typed [`Data`] view, delegating the heavy lifting to [`super::simd`].

use crate::container::{as_bitmap_mut, Bitmap, Data, Run};
use crate::format::*;
use crate::ops::arena::OpArena;
use crate::ops::cursor::{ContainerCursor, ContainerRef};
use crate::ops::plan::{plan_diff, plan_intersect, plan_union, UNION_DENSE_CARD};
use crate::ops::source::Inputs;
use crate::ops::{run, simd};
use crate::{FrozenBitmap, FrozenBitmapView};

/// Drive an op over its keys: at each key, gather the containers present
/// (advancing the cursors) and hand them to `per_key`. Slots are claimed by
/// `per_key`, so keys that produce nothing (e.g. AND misses) cost no slot.
fn fold_keys<I: Inputs + ?Sized>(
    inputs: &I,
    arena: &mut OpArena,
    mut per_key: impl FnMut(&mut OpArena, u16, &[ContainerRef<'_>]),
) {
    let mut cursors: Vec<ContainerCursor<'_>> = (0..inputs.len()).map(|i| inputs.cursor(i)).collect();
    let mut refs: Vec<ContainerRef<'_>> = Vec::with_capacity(inputs.len());
    while let Some(key) = cursors.iter().filter_map(|c| c.peek_key()).min() {
        refs.clear();
        for c in &mut cursors {
            if c.peek_key() == Some(key) {
                refs.push(c.get());
                c.advance();
            }
        }
        per_key(arena, key, &refs);
    }
}

// --- intersection -----------------------------------------------------------

/// N-way intersection (AND). Driven by the input with the fewest containers:
/// only its keys are visited, and the others are `advance_to`-skipped to each —
/// so a selective conjunct never forces a full walk of the large inputs.
pub fn intersect(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    intersect_into(views).serialize()
}

/// AND, folded into a (pooled) arena left for the caller to fold further or
/// serialize — the tree evaluator chains these without a byte round-trip.
pub fn intersect_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let mut arena = OpArena::from_plan(&plan_intersect(inputs));
    if inputs.is_empty() {
        return arena;
    }
    let seed = (0..inputs.len()).min_by_key(|&i| inputs.container_count(i)).unwrap();
    let mut driver = inputs.cursor(seed);
    let mut others: Vec<ContainerCursor<'_>> =
        (0..inputs.len()).filter(|&i| i != seed).map(|i| inputs.cursor(i)).collect();
    let mut refs: Vec<ContainerRef<'_>> = Vec::with_capacity(inputs.len());

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
            let slot = arena.claim();
            intersect_key(&mut arena, slot, key, &refs);
        }
    }
    arena
}

fn intersect_key(arena: &mut OpArena, i: usize, key: u16, refs: &[ContainerRef<'_>]) {
    // Seed from the smallest-card container; its representation fixes the
    // accumulator (array or bitmap) for the whole fold, so it never outgrows
    // the slot. AND only ever shrinks it.
    let seed = (0..refs.len()).min_by_key(|&j| refs[j].card).unwrap();

    if refs[seed].card as usize <= ARRAY_MAX_SIZE {
        let mut card = load_array(arena.slot_mut(i), refs[seed].typed());
        for (j, p) in refs.iter().enumerate() {
            if j == seed || card == 0 {
                continue;
            }
            let (slot, scratch) = arena.slot_and_scratch(i);
            card = array_intersect(slot, card, p.typed(), scratch);
        }
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
    } else if all_runs(refs) && total_runs(refs) <= MAX_RUNS {
        // Dense run containers stay runs (O(runs), not O(bitmap)).
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
        arena.record(key, CT_BITMAP, card, i, BITMAP_BYTES);
    }
}

// --- union ------------------------------------------------------------------

/// N-way union (OR).
pub fn union(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    union_into(views).serialize()
}

/// OR, folded into a (pooled) arena for the caller to chain or serialize.
pub fn union_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let mut arena = OpArena::from_plan(&plan_union(inputs));
    fold_keys(inputs, &mut arena, |arena, key, refs| {
        let slot = arena.claim();
        union_key(arena, slot, key, refs);
    });
    arena
}

fn union_key(arena: &mut OpArena, i: usize, key: u16, refs: &[ContainerRef<'_>]) {
    let sum_card: u32 = refs.iter().map(|p| p.card).fold(0, u32::saturating_add);
    let total_runs: usize = refs.iter().map(|p| p.num_runs()).sum();
    let any_bitmap = refs.iter().any(|p| p.typ == CT_BITMAP);
    let needs_bitmap = any_bitmap || sum_card > UNION_DENSE_CARD || total_runs > MAX_RUNS;

    if needs_bitmap && all_runs(refs) && total_runs <= MAX_RUNS {
        // Run ∪ Run stays a (coalesced) run container.
        let (card, bytes) = run_fold(arena, i, &refs[0], &refs[1..], run::union);
        arena.record(key, CT_RUN, card, i, bytes);
    } else if needs_bitmap {
        let dst = acc(arena.slot_mut(i));
        simd::clear(dst);
        let mut card = 0;
        for (idx, p) in refs.iter().enumerate() {
            if idx + 1 == refs.len() {
                card = or_into_count(dst, p.typed());
            } else {
                or_into(dst, p.typed());
            }
        }
        arena.record(key, CT_BITMAP, card, i, BITMAP_BYTES);
    } else {
        let mut card = load_array(arena.slot_mut(i), refs[0].typed());
        for p in &refs[1..] {
            let (slot, scratch) = arena.slot_and_scratch(i);
            card = array_union(slot, card, p.typed(), scratch);
        }
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
    }
}

// --- difference -------------------------------------------------------------

/// N-way difference: `inputs[0]` minus the rest.
pub fn diff(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    diff_into(views).serialize()
}

/// DIFF, folded into a (pooled) arena for the caller to chain or serialize.
pub fn diff_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let mut arena = OpArena::from_plan(&plan_diff(inputs));
    if inputs.is_empty() {
        return arena;
    }
    let mut a = inputs.cursor(0);
    let mut rhs: Vec<ContainerCursor<'_>> = (1..inputs.len()).map(|i| inputs.cursor(i)).collect();
    let mut refs: Vec<ContainerRef<'_>> = Vec::with_capacity(inputs.len());
    while let Some(key) = a.peek_key() {
        let lhs = a.get();
        a.advance();
        refs.clear();
        for c in &mut rhs {
            if c.advance_to(key) {
                refs.push(c.get());
            }
        }
        let slot = arena.claim();
        diff_key(&mut arena, slot, key, &lhs, &refs);
    }
    arena
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
        let mut card = load_array(arena.slot_mut(i), lhs.typed());
        for p in rhs {
            if card == 0 {
                break;
            }
            let (slot, scratch) = arena.slot_and_scratch(i);
            card = array_diff(slot, card, p.typed(), scratch);
        }
        arena.record(key, CT_ARRAY, card, i, card as usize * 2);
    } else if lhs.typ == CT_RUN
        && all_runs(rhs)
        && lhs.num_runs() + rhs.iter().map(|r| r.num_runs()).sum::<usize>() <= MAX_RUNS
    {
        // Dense run minus dense runs stays a run container.
        let (card, bytes) = run_fold(arena, i, lhs, rhs, run::diff);
        arena.record(key, CT_RUN, card, i, bytes);
    } else {
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
        arena.record(key, CT_BITMAP, card, i, BITMAP_BYTES);
    }
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
fn total_runs(refs: &[ContainerRef<'_>]) -> usize {
    refs.iter().map(|r| r.num_runs()).sum()
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
    op: fn(&[Run], &[Run], &mut [Run]) -> (usize, u32),
) -> (u32, usize) {
    let (slot, scratch) = arena.slot_and_scratch(i);
    let (a, b) = scratch.split_at_mut(BITMAP_BYTES);
    let acc: &mut [Run] = bytemuck::cast_slice_mut(a);
    let tmp: &mut [Run] = bytemuck::cast_slice_mut(b);

    let s = as_runs(seed);
    acc[..s.len()].copy_from_slice(s);
    let mut nr = s.len();
    let mut card = seed.card;
    for p in partners {
        let (n, c) = op(&acc[..nr], as_runs(p), tmp);
        acc[..n].copy_from_slice(&tmp[..n]);
        nr = n;
        card = c;
    }

    write_u16(slot, 0, nr as u16);
    let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut slot[2..2 + nr * 4]);
    dst.copy_from_slice(&acc[..nr]);
    (card, 2 + nr * 4)
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

/// Array accumulator `∩= partner` (in place). `out` aliases the slot; `scratch`
/// stages the accumulator for the two-pointer merge against array partners.
#[inline]
fn array_intersect(slot: &mut [u8], card: u32, data: Data<'_>, scratch: &mut [u8]) -> u32 {
    let acc: &mut [u16] = bytemuck::cast_slice_mut(slot);
    if let Data::Array(b) = data {
        let tmp: &mut [u16] = bytemuck::cast_slice_mut(scratch);
        tmp[..card as usize].copy_from_slice(&acc[..card as usize]);
        return simd::array_intersect(&tmp[..card as usize], b, acc) as u32;
    }
    retain(acc, card, |lo| data.contains(lo))
}

/// Array accumulator `∪= partner` (sorted-merge through `scratch`).
#[inline]
fn array_union(slot: &mut [u8], card: u32, data: Data<'_>, scratch: &mut [u8]) -> u32 {
    let acc: &mut [u16] = bytemuck::cast_slice_mut(slot);
    let (a, b) = scratch.split_at_mut(BITMAP_BYTES);
    let acc_tmp: &mut [u16] = bytemuck::cast_slice_mut(a);
    acc_tmp[..card as usize].copy_from_slice(&acc[..card as usize]);
    let partner: &mut [u16] = bytemuck::cast_slice_mut(b);
    let pn = data.write_sorted(partner);
    simd::array_union(&acc_tmp[..card as usize], &partner[..pn], acc) as u32
}

/// Array accumulator `\= partner` (in place).
#[inline]
fn array_diff(slot: &mut [u8], card: u32, data: Data<'_>, scratch: &mut [u8]) -> u32 {
    let acc: &mut [u16] = bytemuck::cast_slice_mut(slot);
    if let Data::Array(b) = data {
        let tmp: &mut [u16] = bytemuck::cast_slice_mut(scratch);
        tmp[..card as usize].copy_from_slice(&acc[..card as usize]);
        return simd::array_diff(&tmp[..card as usize], b, acc) as u32;
    }
    retain(acc, card, |lo| !data.contains(lo))
}

/// Keep `acc[..card]` values where `keep` holds, compacting in place.
#[inline]
fn retain(acc: &mut [u16], card: u32, mut keep: impl FnMut(u16) -> bool) -> u32 {
    let mut w = 0;
    for r in 0..card as usize {
        let v = acc[r];
        if keep(v) {
            acc[w] = v;
            w += 1;
        }
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
