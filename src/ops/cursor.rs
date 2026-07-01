//! Ascending per-container cursor over a fold input — a byte-encoded leaf
//! (standard or inline) or a working arena. Shared by the analysis pass and the
//! op kernels; mirrors the monorepo `ContainerIter`.

use crate::container::Data;
use crate::format::*;
use crate::ops::arena::OpArena;
use crate::ops::keymask::KeyMask;
use crate::FrozenBitmapView;

/// One container's metadata + payload. For inline inputs, `typ` is
/// [`CT_INLINE`] and `data` is the packed `u32` slice for this key group.
#[derive(Clone, Copy)]
pub struct ContainerRef<'a> {
    pub key: u16,
    pub typ: u8,
    pub card: u32,
    pub data: &'a [u8],
}

impl<'a> ContainerRef<'a> {
    /// Zero-copy typed view of this container's payload.
    #[inline]
    pub fn typed(&self) -> Data<'a> {
        Data::new(self.typ, self.card, self.data)
    }

    /// Bytes this container occupies as a *stored* payload (inline → array form).
    #[inline]
    pub fn stored_bytes(&self) -> usize {
        match self.typ {
            CT_ARRAY => self.card as usize * 2,
            CT_BITMAP => BITMAP_BYTES,
            CT_RUN => self.data.len(),
            CT_INLINE => self.card as usize * 2,
            _ => 0,
        }
    }

    /// Run count, or 0 for non-run containers.
    #[inline]
    pub fn num_runs(&self) -> usize {
        if self.typ == CT_RUN {
            read_u16(self.data, 0) as usize
        } else {
            0
        }
    }
}

/// What a cursor walks: a byte-encoded leaf or a working arena.
enum Backing<'a> {
    /// Standard (SoA index) when `!inline`, else inline (packed `u32`).
    /// `data_base` is the payload/packed-region start.
    Bytes { bytes: &'a [u8], inline: bool, data_base: usize },
    /// An intermediate arena read back in record (key-ascending) order.
    Arena(&'a OpArena),
}

pub struct ContainerCursor<'a> {
    backing: Backing<'a>,
    n: usize,
    // Container index (standard/arena) or value index (inline).
    pos: usize,
    /// Hole-punch mask: when set, the cursor skips every key not in it.
    live: Option<&'a KeyMask>,
}

impl<'a> ContainerCursor<'a> {
    pub fn new(view: &FrozenBitmapView<'a>) -> Self {
        let bytes = view.as_bytes();
        if let Some((n, data_base)) = view.standard_dims() {
            Self { backing: Backing::Bytes { bytes, inline: false, data_base }, n, pos: 0, live: None }
        } else {
            let count = view.inline_count().unwrap_or(0);
            let data_base = INLINE_HEADER_SIZE;
            Self { backing: Backing::Bytes { bytes, inline: true, data_base }, n: count, pos: 0, live: None }
        }
    }

    /// Like [`new`](Self::new), but skips container keys absent from `live`
    /// (hole-punching): the cursor only ever rests on / yields live keys.
    pub fn new_live(view: &FrozenBitmapView<'a>, live: &'a KeyMask) -> Self {
        let mut c = Self::new(view);
        c.live = Some(live);
        c.skip_dead();
        c
    }

    /// Read a working arena as an ordered container source (no serialization).
    pub fn from_arena(arena: &'a OpArena) -> Self {
        debug_assert!(arena.is_key_sorted(), "arena source must be key-ascending");
        Self { backing: Backing::Arena(arena), n: arena.container_count(), pos: 0, live: None }
    }

    /// Key of the current container, or `None` when exhausted.
    #[inline]
    pub fn peek_key(&self) -> Option<u16> {
        if self.pos >= self.n {
            return None;
        }
        Some(match &self.backing {
            Backing::Bytes { bytes, inline: true, data_base } => {
                (read_u32(bytes, data_base + self.pos * 4) >> 16) as u16
            }
            Backing::Bytes { bytes, inline: false, .. } => read_key(bytes, self.n, self.pos),
            Backing::Arena(a) => a.container_key(self.pos),
        })
    }

    /// Current container without advancing.
    pub fn get(&self) -> ContainerRef<'a> {
        debug_assert!(self.pos < self.n);
        match &self.backing {
            Backing::Bytes { bytes, inline: true, data_base } => {
                let key = (read_u32(bytes, data_base + self.pos * 4) >> 16) as u16;
                let mut end = self.pos + 1;
                while end < self.n && (read_u32(bytes, data_base + end * 4) >> 16) as u16 == key {
                    end += 1;
                }
                ContainerRef {
                    key,
                    typ: CT_INLINE,
                    card: (end - self.pos) as u32,
                    data: &bytes[data_base + self.pos * 4..data_base + end * 4],
                }
            }
            Backing::Bytes { bytes, inline: false, data_base } => {
                let e = read_index_entry(bytes, self.n, self.pos);
                let start = data_base + e.data_offset as usize;
                let size = match e.typ {
                    CT_ARRAY => e.cardinality as usize * 2,
                    CT_BITMAP => BITMAP_BYTES,
                    CT_RUN => 2 + read_u16(bytes, start) as usize * 4,
                    _ => 0,
                };
                ContainerRef { key: e.key, typ: e.typ, card: e.cardinality, data: &bytes[start..start + size] }
            }
            Backing::Arena(a) => a.container_ref(self.pos),
        }
    }

    /// Advance past the current container, then rest on the next live key.
    #[inline]
    pub fn advance(&mut self) {
        self.advance_raw();
        if self.live.is_some() {
            self.skip_dead();
        }
    }

    /// Advance past the current container (ignoring the live mask).
    fn advance_raw(&mut self) {
        if let Backing::Bytes { bytes, inline: true, data_base } = &self.backing {
            let Some(key) = self.peek_key() else { return };
            while self.pos < self.n
                && (read_u32(bytes, data_base + self.pos * 4) >> 16) as u16 == key
            {
                self.pos += 1;
            }
        } else {
            self.pos += 1;
        }
    }

    /// Advance past any leading dead (non-live) keys so `pos` rests on a live one.
    fn skip_dead(&mut self) {
        let Some(mask) = self.live else { return };
        while let Some(k) = self.peek_key() {
            if mask.contains(k) {
                break;
            }
            self.advance_raw();
        }
    }

    /// Advance until the current key ≥ `target`. Returns `true` if it equals it.
    /// Dead keys are skipped, so a hit is always a live key.
    pub fn advance_to(&mut self, target: u16) -> bool {
        while self.peek_key().is_some_and(|k| k < target) {
            self.advance_raw();
        }
        let hit = self.peek_key() == Some(target);
        if self.live.is_some() {
            self.skip_dead();
        }
        hit
    }
}

/// The reusable cursor + ref buffers a fold drives with, lifetime-erased for
/// pooling (they are only ever stored empty, see [`FoldScratch`]).
#[derive(Default)]
struct Buffers {
    cursors: Vec<ContainerCursor<'static>>,
    refs: Vec<ContainerRef<'static>>,
}

mod scratch_pool {
    use std::cell::RefCell;

    use super::Buffers;

    const MAX_POOLED: usize = 8;
    thread_local! {
        static POOL: RefCell<Vec<Buffers>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn take() -> Buffers {
        POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
    }

    pub(super) fn put(b: Buffers) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < MAX_POOLED {
                p.push(b);
            }
        });
    }
}

/// Pooled scratch for driving a fold: a cursor buffer and a per-key ref buffer,
/// reused across folds so a fold allocates nothing in steady state. The buffers
/// are loaned at the inputs' lifetime via [`borrow`](Self::borrow) and returned
/// (cleared) on drop — exactly like [`OpArena`]'s working memory.
pub struct FoldScratch {
    cursors: Vec<ContainerCursor<'static>>,
    refs: Vec<ContainerRef<'static>>,
}

impl FoldScratch {
    #[inline]
    pub fn take() -> Self {
        let Buffers { cursors, refs } = scratch_pool::take();
        FoldScratch { cursors, refs }
    }

    /// Borrow the (empty) cursor and ref buffers relabeled to the inputs'
    /// lifetime `'b`.
    #[inline]
    pub fn borrow<'b>(&mut self) -> (&mut Vec<ContainerCursor<'b>>, &mut Vec<ContainerRef<'b>>) {
        debug_assert!(self.cursors.is_empty() && self.refs.is_empty());
        // SAFETY: both buffers are empty across every loan boundary (cleared on
        // take and on drop), so no `'static` cursor/ref is ever materialized —
        // we only relabel the empty buffers to the caller's `'b`. The loaned
        // cursors borrow the fold's inputs and are cleared (on drop) before the
        // fold returns, so they never outlive what they point at. Vec layout is
        // lifetime-invariant, so the relabel is a no-op at runtime.
        unsafe {
            let c = &mut self.cursors as *mut Vec<ContainerCursor<'static>> as *mut Vec<ContainerCursor<'b>>;
            let r = &mut self.refs as *mut Vec<ContainerRef<'static>> as *mut Vec<ContainerRef<'b>>;
            (&mut *c, &mut *r)
        }
    }
}

impl Drop for FoldScratch {
    fn drop(&mut self) {
        self.cursors.clear();
        self.refs.clear();
        scratch_pool::put(Buffers {
            cursors: std::mem::take(&mut self.cursors),
            refs: std::mem::take(&mut self.refs),
        });
    }
}
