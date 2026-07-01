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

use crate::bitmap::{aligned_buf, result_buf, AlignedBuf, FrozenBitmap};
use crate::container::Data;
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
    slot_key: Vec<u16>,
    out: Vec<OutEntry>,
}

impl Default for Reusable {
    fn default() -> Self {
        Self {
            buf: aligned_buf(0),
            slot_off: Vec::new(),
            slot_sz: Vec::new(),
            slot_key: Vec::new(),
            out: Vec::new(),
        }
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
    slot_key: Vec<u16>,
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
            slot_key: std::mem::take(&mut self.slot_key),
            out: std::mem::take(&mut self.out),
        });
    }
}

impl OpArena {
    /// Size a (pooled) buffer for every planned slot + fixed scratch. Bitmap-
    /// capacity slots and the scratch land on 64-byte boundaries for SIMD.
    pub fn from_plan(plan: &Plan) -> Self {
        let Reusable { mut buf, mut slot_off, mut slot_sz, mut slot_key, mut out } = pool::take();
        slot_off.clear();
        slot_sz.clear();
        slot_key.clear();
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
            slot_key,
            out,
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
    /// and each capacity ≥ its payload, so every destination ≤ its source), the
    /// header and index land in the reserved front, and the buffer itself
    /// becomes the result — the fold's writes were the output's writes. A
    /// buffer from the result pool is swapped in so the pools stay balanced.
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
                    let (card, i) = (e.card, e.slot_idx);
                    let bytes = card as usize * 2;
                    let (slot, scratch) = self.slot_and_scratch(i);
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
        // and slots were claimed in key order, so source offsets ascend too and
        // never precede their destinations.
        let (mut dc, mut prev_end) = (0usize, data_base);
        for j in 0..n {
            let e = &self.out[j];
            let (key, typ, card, size) = (e.key, e.typ, e.card, e.data_size as usize);
            let src = self.slot_off[e.slot_idx] as usize;
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
