//! Machinery every fold shares: the tiny-input gate, the key-major driver, the
//! array/bitmap/run accumulators, and the slot load/store helpers.

use crate::api::container::{as_bitmap_mut, Bitmap, Data, Run};
use crate::format::*;
use crate::ops::arena::{OpArena, SlotState};
use crate::ops::cursor::{ContainerRef, FoldScratch};
use crate::ops::source::Inputs;
use crate::ops::simd;
use crate::FrozenBitmapView;

// --- key-major fold order (cache-heavy accumulator sets) ---------------------

/// Drive an op over its keys: at each key, gather the containers present
/// (advancing the cursors) and hand them to `per_key`. Slots are claimed by
/// `per_key`, so keys that produce nothing (e.g. AND misses) cost no slot.
/// Tiny-input one-shot gate: below this, the capacity walk dominates the fold
/// (run cells carry ~18-byte payloads), so ∧/∖ skip it via [`plan_trivial`].
/// The count cap bounds the trivial arena (`keys × B`).
const TRIVIAL_MAX_BYTES: usize = 16 << 10;
const TRIVIAL_MAX_KEYS: usize = 32;

pub(super) fn trivial(views: &[FrozenBitmapView<'_>], drive: usize) -> bool {
    views[drive].num_containers() <= TRIVIAL_MAX_KEYS
        && views.iter().map(|v| v.as_bytes().len()).sum::<usize>() <= TRIVIAL_MAX_BYTES
}

pub(super) fn fold_keys<I: Inputs + ?Sized>(
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


// --- shared helpers ---------------------------------------------------------

/// Whether every container is run-encoded — the precondition for a native run
/// fold. Taken only inside a bitmap-sized slot, where a `≤ MAX_RUNS` run
/// container always fits.
#[inline]
pub(super) fn all_runs(refs: &[ContainerRef<'_>]) -> bool {
    !refs.is_empty() && refs.iter().all(|r| r.typ == CT_RUN)
}

#[inline]
pub(super) fn as_runs<'a>(r: &ContainerRef<'a>) -> &'a [Run] {
    match r.typed() {
        Data::Run(runs) => runs,
        _ => unreachable!("as_runs on a non-run container"),
    }
}

/// Fold `seed` then `partners` with a native run `op`, writing a `CT_RUN`
/// container into slot `i`. Scratch is split into two run buffers and the
/// result copied to the slot. Returns `(cardinality, data bytes)`.
pub(super) fn run_fold(
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
    slot[2..run_bytes(nr)].copy_from_slice(&cur[..nr * 4]);
    (card, run_bytes(nr))
}

/// Finish a bitmap accumulator: downgrade to an array when sparse enough
/// (cheaper downstream folds + smaller output), else keep the bitmap. The
/// boundary is half the array limit: above it, extraction (~0.5 ns/value)
/// buys at most a 2x-shrinking output while costing more than the fold that
/// produced it — a card-3900 extraction was 63% of a dense 4-way difference.
/// `serialize_compact` still canonicalizes terminal results. Returns the
/// recorded `(type, data bytes)`.
pub(super) fn finish_bitmap(arena: &mut OpArena, i: usize, card: u32) -> (u8, usize) {
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
pub(super) fn acc(slot: &mut [u8]) -> &mut Bitmap {
    as_bitmap_mut(&mut slot[..BITMAP_BYTES])
}

/// Write a container's lows into `slot` as a sorted `u16` array; returns count.
#[inline]
pub(super) fn load_array(slot: &mut [u8], data: Data<'_>) -> u32 {
    data.write_sorted(bytemuck::cast_slice_mut(slot)) as u32
}

/// Populate `slot` as a bitmap from `data`. A bitmap source is copied in one
/// pass; others clear then set.
#[inline]
pub(super) fn load_bitmap(slot: &mut [u8], data: Data<'_>) {
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
pub(super) fn or_into(dst: &mut Bitmap, data: Data<'_>) {
    match data {
        Data::Array(a) => simd::set_values(dst, a),
        Data::Run(r) => simd::set_runs(dst, r),
        Data::Bitmap(b) => simd::or(dst, b),
        Data::Inline(ids) => ids.iter().for_each(|&v| set_bit(dst, v as u16)),
    }
}

/// `dst &= !data`.
#[inline]
pub(super) fn clear_into(dst: &mut Bitmap, data: Data<'_>) {
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
pub(super) fn bitmap_and_count(dst: &mut Bitmap, scratch: &mut [u8], data: Data<'_>) -> u32 {
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
pub(super) fn or_into_count(dst: &mut Bitmap, data: Data<'_>) -> u32 {
    if let Data::Bitmap(b) = data {
        simd::or_count(dst, b)
    } else {
        or_into(dst, data);
        simd::popcount(dst)
    }
}

/// `clear_into` fused with a population count.
#[inline]
pub(super) fn clear_into_count(dst: &mut Bitmap, data: Data<'_>) -> u32 {
    if let Data::Bitmap(b) = data {
        simd::andnot_count(dst, b)
    } else {
        clear_into(dst, data);
        simd::popcount(dst)
    }
}

/// The slot's bytes as a `u16` merge-output buffer.
#[inline]
pub(super) fn acc_u16(slot: &mut [u8]) -> &mut [u16] {
    bytemuck::cast_slice_mut(slot)
}

/// Array accumulator for one key's fold. Array×array merges ping-pong between
/// the slot and the first scratch half (the merge kernels need `out` disjoint
/// from both inputs, so flipping sides replaces a per-fold staging copy); run
/// and bitmap partners filter in place. [`finish`](Self::finish) copies back at
/// most once, whatever the arity.
pub(super) struct ArrayAcc {
    pub(super) card: u32,
    in_scratch: bool,
}

impl ArrayAcc {
    pub(super) fn new() -> Self {
        ArrayAcc { card: 0, in_scratch: false }
    }

    /// AND (`keep`) / DIFF (`!keep`) fold with one partner.
    pub(super) fn fold(&mut self, arena: &mut OpArena, i: usize, partner: Data<'_>, keep: bool) {
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
    pub(super) fn fold_union(&mut self, arena: &mut OpArena, i: usize, partner: Data<'_>) {
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
    pub(super) fn finish(self, arena: &mut OpArena, i: usize) -> u32 {
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
pub(super) fn retain_runs(acc: &mut [u16], card: u32, runs: &[Run], keep_inside: bool) -> u32 {
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

/// Keep `acc[..card]` values by bitmap membership (`keep_inside` selects in- vs
/// out-of-set), compacting in place. The word array is hoisted out of the loop
/// and compaction is branchless (~1 ns/value) — versus a per-probe container-enum
/// re-match (~3 ns) — since this is the hot path for a bitmap-partner array fold.
#[inline]
pub(super) fn retain_bitmap(acc: &mut [u16], card: u32, b: &Bitmap, keep_inside: bool) -> u32 {
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
pub(super) fn set_bit(dst: &mut Bitmap, lo: u16) {
    dst[lo as usize / 64] |= 1u64 << (lo as usize % 64);
}

#[inline]
pub(super) fn clear_bit(dst: &mut Bitmap, lo: u16) {
    dst[lo as usize / 64] &= !(1u64 << (lo as usize % 64));
}

#[inline]
pub(super) fn set_state(arena: &mut OpArena, i: usize, typ: u8, runs: u16, card: u32) {
    let side = arena.state(i).side;
    *arena.state_mut(i) = SlotState { typ, side, runs, card };
}

/// Runs held in a slot, in wire layout (`u16` count + `(start, len)` pairs).
#[inline]
pub(super) fn slot_runs(slot: &[u8], nr: u16) -> &[Run] {
    bytemuck::cast_slice(&slot[2..run_bytes(nr as usize)])
}

/// Scatter an array accumulator into bitmap form on the other side.
pub(super) fn array_acc_to_bitmap(arena: &mut OpArena, s: usize) {
    let st = arena.state(s);
    let (cur, other) = arena.slot_pair(s);
    let dst = as_bitmap_mut(&mut other[..BITMAP_BYTES]);
    simd::clear(dst);
    simd::set_values(dst, &bytemuck::cast_slice(cur)[..st.card as usize]);
    arena.flip_side(s);
    set_state(arena, s, CT_BITMAP, 0, st.card);
}

/// Expand a run accumulator into bitmap form on the other side.
pub(super) fn run_acc_to_bitmap(arena: &mut OpArena, s: usize) {
    let st = arena.state(s);
    let (cur, other) = arena.slot_pair(s);
    let dst = as_bitmap_mut(&mut other[..BITMAP_BYTES]);
    simd::clear(dst);
    simd::set_runs(dst, slot_runs(cur, st.runs));
    arena.flip_side(s);
    set_state(arena, s, CT_BITMAP, 0, st.card);
}
