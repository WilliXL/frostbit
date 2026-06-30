//! Working arena for a set op: sized once from a [`Plan`], never grows.
//!
//! There is deliberately no slot-grow or realloc path — the analysis pass sized
//! every slot to a proven ceiling, so kernels only ever write into pre-existing
//! slots. `record` debug-asserts each container fits its slot, turning a
//! planner bug into a test failure rather than a silent runtime allocation.
//!
//! The working buffer and index vectors are reused across ops via a small
//! thread-local pool (see [`pool`]), so steady-state `intersect`/`union`/`diff`
//! allocate only their result.

use crate::bitmap::{aligned_buf, AlignedBuf, FrozenBitmap};
use crate::format::*;
use crate::ops::cursor::ContainerRef;
use crate::ops::plan::Plan;

struct OutEntry {
    key: u16,
    typ: u8,
    card: u32,
    slot_idx: usize,
    data_size: u32,
}

/// The poolable, reusable allocations of an arena.
struct Reusable {
    buf: AlignedBuf,
    slot_off: Vec<u32>,
    slot_sz: Vec<u32>,
    out: Vec<OutEntry>,
}

impl Default for Reusable {
    fn default() -> Self {
        Self { buf: aligned_buf(0), slot_off: Vec::new(), slot_sz: Vec::new(), out: Vec::new() }
    }
}

/// Thread-local reuse of arena working memory. A small stack handles the
/// nesting that the (future) expression-tree evaluator introduces.
mod pool {
    use std::cell::RefCell;

    use super::Reusable;

    const MAX_POOLED: usize = 8;
    thread_local! {
        static POOL: RefCell<Vec<Reusable>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn take() -> Reusable {
        POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
    }

    pub(super) fn put(r: Reusable) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < MAX_POOLED {
                p.push(r);
            }
        });
    }
}

pub struct OpArena {
    buf: AlignedBuf,
    slot_off: Vec<u32>,
    slot_sz: Vec<u32>,
    out: Vec<OutEntry>,
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
            out: std::mem::take(&mut self.out),
        });
    }
}

impl OpArena {
    /// Size a (pooled) buffer for every planned slot + fixed scratch. Bitmap-
    /// capacity slots and the scratch land on 64-byte boundaries for SIMD.
    pub fn from_plan(plan: &Plan) -> Self {
        let Reusable { mut buf, mut slot_off, mut slot_sz, mut out } = pool::take();
        slot_off.clear();
        slot_sz.clear();
        out.clear();

        let mut cursor = 0usize;
        for s in &plan.slots {
            let align = if s.capacity as usize == BITMAP_BYTES { BUF_ALIGN } else { WORD_ALIGN };
            cursor = align_up(cursor, align);
            slot_off.push(cursor as u32);
            slot_sz.push(s.capacity);
            cursor += s.capacity as usize;
        }
        let scratch_off = align_up(cursor, BUF_ALIGN);
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
            out,
            scratch_off,
            next_slot: 0,
            total_card: 0,
            has_runs: false,
            has_bitmap: false,
        }
    }

    /// Hand out the next slot index in plan order. Slots are claimed only when
    /// actually produced (e.g. AND skips non-shared keys), so the count matches
    /// the plan exactly — there is no grow path if this is exceeded.
    #[inline]
    pub fn claim(&mut self) -> usize {
        let i = self.next_slot;
        debug_assert!(i < self.num_slots(), "claimed more slots than planned");
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

    /// Writable bytes of slot `i` (full capacity).
    #[inline]
    pub fn slot_mut(&mut self, i: usize) -> &mut [u8] {
        let off = self.slot_off[i] as usize;
        let sz = self.slot_sz[i] as usize;
        &mut self.buf[off..off + sz]
    }

    /// Slot `i` and the scratch region as disjoint mutable slices.
    #[inline]
    pub fn slot_and_scratch(&mut self, i: usize) -> (&mut [u8], &mut [u8]) {
        let off = self.slot_off[i] as usize;
        let sz = self.slot_sz[i] as usize;
        let (front, scratch) = self.buf.split_at_mut(self.scratch_off);
        (&mut front[off..off + sz], scratch)
    }

    /// Record a produced container in slot `i`. Skips empty results.
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
        self.out.push(OutEntry { key, typ, card, slot_idx: i, data_size: data_size as u32 });
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    #[inline]
    pub fn total_cardinality(&self) -> u64 {
        self.total_card
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
        let off = self.slot_off[e.slot_idx] as usize;
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

    /// Compact recorded containers into a standard-format [`FrozenBitmap`].
    pub fn serialize(mut self) -> FrozenBitmap {
        self.out.sort_unstable_by_key(|e| e.key);
        let n = self.out.len();
        let data_base = data_section_off(n, self.has_bitmap);

        // Output payload offsets: bitmaps 64-aligned, else 2-aligned (matches
        // the builder, so arena output is byte-identical for the same set).
        let mut offsets = Vec::with_capacity(n);
        let mut dc = 0usize;
        for e in &self.out {
            let align = if e.typ == CT_BITMAP { BUF_ALIGN } else { 2 };
            dc = align_up(dc, align);
            offsets.push(dc);
            dc += e.data_size as usize;
        }
        let total = data_base + dc;

        // The result buffer is filled completely below — header, index, the
        // alignment gaps, and every payload — so we skip the initial memset.
        // SAFETY: `u8` needs no init and no byte is read before being written.
        let mut buf = aligned_buf(total);
        unsafe { buf.set_len(total) };

        Header {
            has_runs: self.has_runs,
            has_bitmap: self.has_bitmap,
            num_containers: n as u32,
            cardinality: self.total_card,
        }
        .write(&mut buf);

        for (i, e) in self.out.iter().enumerate() {
            write_index_entry(
                &mut buf,
                n,
                i,
                IndexEntry { key: e.key, typ: e.typ, cardinality: e.card, data_offset: offsets[i] as u32 },
            );
        }

        // Zero the gap between the index and the data section, then copy each
        // payload, zeroing the alignment padding that precedes it. This leaves
        // the output byte-identical to a zero-initialized buffer.
        buf[HEADER_SIZE + index_size(n)..data_base].fill(0);
        let mut written = 0usize;
        for (i, e) in self.out.iter().enumerate() {
            let off = offsets[i];
            buf[data_base + written..data_base + off].fill(0);
            let src = self.slot_off[e.slot_idx] as usize;
            let sz = e.data_size as usize;
            buf[data_base + off..data_base + off + sz].copy_from_slice(&self.buf[src..src + sz]);
            written = off + sz;
        }
        buf[data_base + written..total].fill(0);
        FrozenBitmap::from_buf(buf)
    }
}
