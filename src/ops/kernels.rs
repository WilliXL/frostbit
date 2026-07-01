//! Fold kernels. Each writes only into pre-sized arena slots (the arena's
//! `record` debug-asserts the no-runtime-allocation invariant) and dispatches
//! on the typed [`Data`] view, delegating the heavy lifting to [`super::simd`].

use crate::container::{as_bitmap_mut, Bitmap, Data, Run};
use crate::format::*;
use crate::ops::arena::OpArena;
use crate::ops::cursor::{ContainerRef, FoldScratch};
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
    intersect_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (AND) into a pre-sized `arena`. The arena's slots come from the
/// manifest, so this does no sizing analysis — it only drives keys and folds.
pub fn intersect_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    if inputs.is_empty() {
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

fn intersect_key(arena: &mut OpArena, i: usize, key: u16, refs: &[ContainerRef<'_>]) {
    // Seed from the smallest-card container; its representation fixes the
    // accumulator (array or bitmap) for the whole fold, so it never outgrows
    // the slot. AND only ever shrinks it.
    let seed = (0..refs.len()).min_by_key(|&j| refs[j].card).unwrap();

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
    let mut arena = OpArena::from_plan(&plan_union(inputs));
    union_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (OR) into a pre-sized `arena` (no sizing analysis).
pub fn union_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
    fold_keys(inputs, arena, |arena, key, refs| {
        let slot = arena.claim_key(key);
        union_key(arena, slot, key, refs);
    });
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

// --- difference -------------------------------------------------------------

/// N-way difference: `inputs[0]` minus the rest.
pub fn diff(views: &[FrozenBitmapView<'_>]) -> FrozenBitmap {
    diff_into(views).serialize()
}

/// DIFF, folded into a (pooled) arena for the caller to chain or serialize.
pub fn diff_into<I: Inputs + ?Sized>(inputs: &I) -> OpArena {
    let mut arena = OpArena::from_plan(&plan_diff(inputs));
    diff_fold(&mut arena, inputs);
    arena
}

/// Fold `inputs` (DIFF: `inputs[0]` minus the rest) into a pre-sized `arena`.
pub fn diff_fold<I: Inputs + ?Sized>(arena: &mut OpArena, inputs: &I) {
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
    let acc: &mut [Run] = bytemuck::cast_slice_mut(a);
    let tmp: &mut [Run] = bytemuck::cast_slice_mut(b);

    let l = as_runs(lhs);
    acc[..l.len()].copy_from_slice(l);
    let mut nr = l.len();
    let mut card = lhs.card;
    for p in rhs {
        let (n, c) = match p.typed() {
            Data::Run(pr) => run::diff(&acc[..nr], pr, tmp),
            Data::Array(pa) => run::diff_array(&acc[..nr], pa, tmp),
            _ => unreachable!("run_fold_diff partner must be run or array"),
        };
        acc[..n].copy_from_slice(&tmp[..n]);
        nr = n;
        card = c;
    }

    write_u16(slot, 0, nr as u16);
    let dst: &mut [Run] = bytemuck::cast_slice_mut(&mut slot[2..2 + nr * 4]);
    dst.copy_from_slice(&acc[..nr]);
    (card, 2 + nr * 4)
}

/// Finish a bitmap accumulator: downgrade to an array when sparse enough
/// (cheaper downstream folds + smaller output), else keep the bitmap. Returns
/// the recorded `(type, data bytes)`.
fn finish_bitmap(arena: &mut OpArena, i: usize, card: u32) -> (u8, usize) {
    if card == 0 || card > ARRAY_MAX_SIZE as u32 {
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
        let (sa, _) = scratch.split_at_mut(BITMAP_BYTES);
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
            _ => {
                let cur = if self.in_scratch { sa } else { slot };
                self.card = retain(acc_u16(cur), self.card, |lo| partner.contains(lo) == keep);
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
