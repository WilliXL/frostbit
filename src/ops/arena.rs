//! Working arena for a set op: sized once from a [`Plan`], never grows.
//!
//! There is deliberately no slot-grow or realloc path — the analysis pass sized
//! every slot to a proven ceiling, so kernels only ever write into pre-existing
//! slots. `record` debug-asserts each container fits its slot, turning a
//! planner bug into a test failure rather than a silent runtime allocation.
//!
//! A double-buffered plan (see [`Plan::double`]) lays a mirror region after the
//! slots: partner-major folds flip each slot between its two sides so array
//! merges get `out ≠ in` with no staging copy, while every pass streams its
//! input leaf sequentially.
//!
//! The working buffer and index vectors are reused across ops via a small
//! thread-local `pool`, so steady-state `intersect`/`union`/`diff` allocate only
//! their result.

use crate::bitmap::{aligned_buf, result_buf, AlignedBuf, FrozenBitmap};
use crate::container::Data;
use crate::format::*;
use crate::ops::cursor::ContainerRef;
use crate::ops::plan::Plan;

struct OutEntry {
    key: u16,
    typ: u8,
    card: u32,
    /// Side-resolved payload offset in the arena buffer.
    off: u32,
    data_size: u32,
}

/// Per-slot accumulator state for a partner-major fold: the container form the
/// slot currently holds, which side of the double buffer it lives on, and its
/// running run-count / cardinality. `typ == UNSEEDED` marks untouched slots;
/// `card == CARD_LAZY` marks a bitmap whose count is deferred to finalize.
#[derive(Clone, Copy)]
pub(crate) struct SlotState {
    pub typ: u8,
    pub side: u8,
    pub runs: u16,
    pub card: u32,
}

impl SlotState {
    pub const UNSEEDED: u8 = u8::MAX;
    pub const CARD_LAZY: u32 = u32::MAX;

    #[inline]
    pub fn seeded(&self) -> bool {
        self.typ != Self::UNSEEDED
    }
}

/// The poolable, reusable allocations of an arena.
struct Reusable {
    buf: AlignedBuf,
    slot_off: Vec<u32>,
    slot_sz: Vec<u32>,
    slot_key: Vec<u16>,
    state: Vec<SlotState>,
    out: Vec<OutEntry>,
}

impl Default for Reusable {
    fn default() -> Self {
        Self::with_bytes(0)
    }
}

impl Reusable {
    /// Working memory with `bytes` of buffer capacity reserved up front.
    fn with_bytes(bytes: usize) -> Self {
        Self {
            buf: aligned_buf(bytes),
            slot_off: Vec::new(),
            slot_sz: Vec::new(),
            slot_key: Vec::new(),
            state: Vec::new(),
            out: Vec::new(),
        }
    }
}

/// Thread-local reuse of arena working memory. A small stack handles the
/// nesting that the expression-tree evaluator introduces.
mod pool {
    use super::Reusable;
    use crate::pool::Pool;

    thread_local! {
        static POOL: Pool<Reusable> = const { Pool::new("arena") };
    }

    pub(super) fn take() -> Reusable {
        POOL.with(|p| p.take(Reusable::default))
    }

    pub(super) fn put(r: Reusable) {
        POOL.with(|p| p.put(r));
    }

    pub(crate) fn prewarm(sizes: &[usize]) {
        POOL.with(|p| p.prewarm(sizes, Reusable::with_bytes));
    }

    pub(crate) fn clear() {
        POOL.with(Pool::clear);
    }

    pub(crate) fn stats() -> (usize, usize, usize) {
        POOL.with(|p| p.stats(|r| r.buf.capacity()))
    }
}

pub(crate) use pool::{
    clear as clear_arena_pool, prewarm as prewarm_arena_pool, stats as arena_pool_stats,
};

pub struct OpArena {
    buf: AlignedBuf,
    slot_off: Vec<u32>,
    slot_sz: Vec<u32>,
    slot_key: Vec<u16>,
    state: Vec<SlotState>,
    out: Vec<OutEntry>,
    /// Byte distance from a slot's side-A offset to its side-B mirror (0 when
    /// the plan is single-buffered).
    stride: usize,
    scratch_off: usize,
    next_slot: usize,
    total_card: u64,
    has_runs: bool,
    has_bitmap: bool,
}

impl Drop for OpArena {
    fn drop(&mut self) {
        pool::put(Reusable {
            buf: std::mem::replace(&mut self.buf, aligned_buf(0)),
            slot_off: std::mem::take(&mut self.slot_off),
            slot_sz: std::mem::take(&mut self.slot_sz),
            slot_key: std::mem::take(&mut self.slot_key),
            state: std::mem::take(&mut self.state),
            out: std::mem::take(&mut self.out),
        });
    }
}

impl OpArena {
    /// Size a (pooled) buffer for every planned slot + fixed scratch. Bitmap-
    /// capacity slots and the scratch land on 64-byte boundaries for SIMD.
    pub fn from_plan(plan: &Plan) -> Self {
        let Reusable { mut buf, mut slot_off, mut slot_sz, mut slot_key, mut state, mut out } =
            pool::take();
        slot_off.clear();
        slot_sz.clear();
        slot_key.clear();
        state.clear();
        out.clear();

        // Reserve worst-case header + index room up front, so `serialize` can
        // compact payloads leftward in place and emit this very buffer as the
        // result — the fold writes directly into what becomes the output.
        let mut cursor = data_section_off(plan.num_slots(), true);
        for s in &plan.slots {
            let align = if s.capacity as usize == BITMAP_BYTES { BUF_ALIGN } else { WORD_ALIGN };
            cursor = align_up(cursor, align);
            slot_off.push(cursor as u32);
            slot_sz.push(s.capacity);
            slot_key.push(s.key);
            cursor += s.capacity as usize;
        }
        state.resize(
            plan.num_slots(),
            SlotState { typ: SlotState::UNSEEDED, side: 0, runs: 0, card: 0 },
        );
        // A 64-byte-aligned stride keeps side-B bitmap slots cache-aligned too.
        let stride = if plan.double { align_up(cursor, BUF_ALIGN) } else { 0 };
        let scratch_off = align_up(cursor + stride, BUF_ALIGN);
        let total = scratch_off + plan.scratch_bytes;

        // Reuse the pooled buffer's capacity, leaving the bytes uninitialized.
        // SAFETY: `u8` needs no initialization, and every byte a kernel later
        // reads is written first — slots are fully populated before `record`,
        // scratch is cleared before use, and `serialize` only reads recorded
        // `[..data_size]` ranges. So skipping the memset (≈half of a dense op,
        // per profiling) is sound.
        buf.clear();
        buf.reserve(total);
        unsafe { buf.set_len(total) };
        Self {
            buf,
            slot_off,
            slot_sz,
            slot_key,
            state,
            out,
            stride,
            scratch_off,
            next_slot: 0,
            total_card: 0,
            has_runs: false,
            has_bitmap: false,
        }
    }

    /// Hand out the slot planned for `key`, skipping planned keys that produced
    /// nothing (the manifest's key set is a superset of what the fold emits, so
    /// claiming by key — not position — lands on the correctly-sized slot).
    /// Keys are claimed ascending, so this is amortized O(1).
    #[inline]
    pub fn claim_key(&mut self, key: u16) -> usize {
        while self.slot_key[self.next_slot] < key {
            self.next_slot += 1;
        }
        debug_assert_eq!(self.slot_key[self.next_slot], key, "claimed a key absent from the plan");
        let i = self.next_slot;
        self.next_slot += 1;
        i
    }

    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slot_sz.len()
    }

    #[inline]
    pub fn slot_capacity(&self, i: usize) -> usize {
        self.slot_sz[i] as usize
    }

    /// The planned key of slot `i` (partner-major passes pair cursors to it).
    #[inline]
    pub(crate) fn planned_key(&self, i: usize) -> u16 {
        self.slot_key[i]
    }

    #[inline]
    pub(crate) fn state(&self, i: usize) -> SlotState {
        self.state[i]
    }

    #[inline]
    pub(crate) fn state_mut(&mut self, i: usize) -> &mut SlotState {
        &mut self.state[i]
    }

    /// Byte offset of slot `i`'s current side.
    #[inline]
    fn cur_off(&self, i: usize) -> usize {
        self.slot_off[i] as usize + self.state[i].side as usize * self.stride
    }

    /// Writable bytes of slot `i` (full capacity, current side).
    #[inline]
    pub fn slot_mut(&mut self, i: usize) -> &mut [u8] {
        let off = self.cur_off(i);
        let sz = self.slot_sz[i] as usize;
        &mut self.buf[off..off + sz]
    }

    /// Slot `i` (current side) and the scratch region as disjoint mut slices.
    #[inline]
    pub fn slot_and_scratch(&mut self, i: usize) -> (&mut [u8], &mut [u8]) {
        let off = self.cur_off(i);
        let sz = self.slot_sz[i] as usize;
        let (front, scratch) = self.buf.split_at_mut(self.scratch_off);
        (&mut front[off..off + sz], scratch)
    }

    /// Slot `i`'s two sides as `(current, other)` disjoint mut slices, for a
    /// merge whose output must not alias its input. Flip with
    /// [`flip_side`](Self::flip_side) after writing `other`.
    #[inline]
    pub(crate) fn slot_pair(&mut self, i: usize) -> (&mut [u8], &mut [u8]) {
        debug_assert!(self.stride > 0, "slot_pair needs a double-buffered plan");
        let a = self.slot_off[i] as usize;
        let b = a + self.stride;
        let sz = self.slot_sz[i] as usize;
        let (lo, hi) = self.buf.split_at_mut(b);
        let (a_sl, b_sl) = (&mut lo[a..a + sz], &mut hi[..sz]);
        if self.state[i].side == 0 {
            (a_sl, b_sl)
        } else {
            (b_sl, a_sl)
        }
    }

    /// Slot `i`'s two sides plus the scratch region, all disjoint — for a merge
    /// whose partner must be staged (extracted) before merging.
    #[inline]
    pub(crate) fn slot_pair_and_scratch(
        &mut self,
        i: usize,
    ) -> (&mut [u8], &mut [u8], &mut [u8]) {
        debug_assert!(self.stride > 0, "slot_pair needs a double-buffered plan");
        let a = self.slot_off[i] as usize;
        let b = a + self.stride;
        let sz = self.slot_sz[i] as usize;
        let (front, scratch) = self.buf.split_at_mut(self.scratch_off);
        let (lo, hi) = front.split_at_mut(b);
        let (a_sl, b_sl) = (&mut lo[a..a + sz], &mut hi[..sz]);
        if self.state[i].side == 0 {
            (a_sl, b_sl, scratch)
        } else {
            (b_sl, a_sl, scratch)
        }
    }

    /// Whether this arena carries the mirror region (partner-major dispatch).
    #[inline]
    pub(crate) fn is_double(&self) -> bool {
        self.stride > 0
    }

    #[inline]
    pub(crate) fn flip_side(&mut self, i: usize) {
        self.state[i].side ^= 1;
    }

    /// Record a produced container in slot `i` (current side). Skips empties.
    pub fn record(&mut self, key: u16, typ: u8, card: u32, i: usize, data_size: usize) {
        if card == 0 {
            return;
        }
        debug_assert!(
            data_size <= self.slot_sz[i] as usize,
            "slot {i} overflow: {data_size} > capacity {} (key {key})",
            self.slot_sz[i]
        );
        self.has_runs |= typ == CT_RUN;
        self.has_bitmap |= typ == CT_BITMAP;
        self.total_card += card as u64;
        let off = self.cur_off(i) as u32;
        self.out.push(OutEntry { key, typ, card, off, data_size: data_size as u32 });
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    /// Number of recorded containers — an arena read back as a container source.
    #[inline]
    pub fn container_count(&self) -> usize {
        self.out.len()
    }

    /// Key of the `i`-th recorded container (cheap cursor peek).
    #[inline]
    pub(crate) fn container_key(&self, i: usize) -> u16 {
        self.out[i].key
    }

    /// The `i`-th recorded container as a zero-copy [`ContainerRef`]. Records are
    /// ascending by key (every kernel drives its keys ascending), so an arena
    /// reads back as an ordered source — no re-sort, no serialization.
    #[inline]
    pub(crate) fn container_ref(&self, i: usize) -> ContainerRef<'_> {
        let e = &self.out[i];
        let off = e.off as usize;
        ContainerRef {
            key: e.key,
            typ: e.typ,
            card: e.card,
            data: &self.buf[off..off + e.data_size as usize],
        }
    }

    /// Invariant for reading an arena as an ordered source (checked in debug).
    pub(crate) fn is_key_sorted(&self) -> bool {
        self.out.windows(2).all(|w| w[0].key <= w[1].key)
    }

    /// Serialize recorded containers as op-ready (`_fast`) output — verbatim, so
    /// the bytes are identical to the builder's for the same set.
    pub fn serialize(self) -> FrozenBitmap {
        self.serialize_inner(false)
    }

    /// Serialize smallest (`_compact`): a sparse bitmap (card ≤ [`ARRAY_MAX_SIZE`])
    /// is transcoded to an array, so a terminal result is as small as roaring's.
    pub fn serialize_compact(self) -> FrozenBitmap {
        self.serialize_inner(true)
    }

    /// Serialize **in place**: payloads are compacted leftward inside the
    /// arena's own buffer (slots were laid past a reserved header+index front,
    /// and each capacity ≥ its payload, so every destination ≤ its source —
    /// side-B payloads sit past the whole side-A region, so the bound holds for
    /// them a fortiori), the header and index land in the reserved front, and
    /// the buffer itself becomes the result — the fold's writes were the
    /// output's writes. A buffer from the result pool is swapped in so the
    /// pools stay balanced.
    fn serialize_inner(mut self, compact: bool) -> FrozenBitmap {
        self.out.sort_unstable_by_key(|e| e.key);
        let n = self.out.len();

        // Compact transcodes (sparse bitmap → array) bounce through scratch
        // first — an overlapping in-place bit extraction would clobber unread
        // words. Sizes only shrink, preserving the compaction invariant.
        if compact {
            for j in 0..n {
                let e = &self.out[j];
                if e.typ == CT_BITMAP && e.card as usize <= ARRAY_MAX_SIZE {
                    let (card, off) = (e.card, e.off as usize);
                    let bytes = card as usize * 2;
                    let (front, scratch) = self.buf.split_at_mut(self.scratch_off);
                    let slot = &mut front[off..off + BITMAP_BYTES];
                    let w = Data::new(CT_BITMAP, card, &slot[..BITMAP_BYTES])
                        .write_sorted(bytemuck::cast_slice_mut(scratch));
                    debug_assert_eq!(w, card as usize);
                    slot[..bytes].copy_from_slice(&scratch[..bytes]);
                    let e = &mut self.out[j];
                    e.typ = CT_ARRAY;
                    e.data_size = bytes as u32;
                }
            }
        }

        let (mut has_bitmap, mut has_runs) = (false, false);
        for e in &self.out {
            has_bitmap |= e.typ == CT_BITMAP;
            has_runs |= e.typ == CT_RUN;
        }
        let data_base = data_section_off(n, has_bitmap);

        // Compact payloads leftward, writing each index entry as its payload
        // lands and zeroing the alignment gap before it. Records ascend by key
        // and slots were claimed in key order, so source offsets never precede
        // their destinations.
        let (mut dc, mut prev_end) = (0usize, data_base);
        for j in 0..n {
            let e = &self.out[j];
            let (key, typ, card, size) = (e.key, e.typ, e.card, e.data_size as usize);
            let src = e.off as usize;
            let align = if typ == CT_BITMAP { BUF_ALIGN } else { 2 };
            dc = align_up(dc, align);
            let dst = data_base + dc;
            debug_assert!(dst <= src, "in-place compaction must move left");
            let entry = IndexEntry { key, typ, cardinality: card, data_offset: dc as u32 };
            write_index_entry(&mut self.buf, n, j, entry);
            self.buf[prev_end..dst].fill(0);
            if dst != src {
                self.buf.copy_within(src..src + size, dst);
            }
            prev_end = dst + size;
            dc += size;
        }
        let total = data_base + dc;

        Header {
            has_runs,
            has_bitmap,
            num_containers: n as u32,
            cardinality: self.total_card,
        }
        .write(&mut self.buf);
        self.buf[HEADER_SIZE + index_size(n)..data_base].fill(0);
        self.buf.truncate(total);

        // Hand this buffer out as the result; swap in one from the result pool
        // so the arena pool gets a buffer back (they circulate, no malloc).
        let buf = std::mem::replace(&mut self.buf, result_buf(0));
        FrozenBitmap::from_buf(buf)
    }
}
