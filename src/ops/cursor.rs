//! Ascending per-container cursor over a fold input — a byte-encoded leaf
//! (standard or inline) or a working arena. Shared by the analysis pass and the
//! op kernels; mirrors the monorepo `ContainerIter`.

use crate::container::Data;
use crate::format::*;
use crate::ops::arena::OpArena;
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
}

impl<'a> ContainerCursor<'a> {
    pub fn new(view: &FrozenBitmapView<'a>) -> Self {
        let bytes = view.as_bytes();
        if let Some((n, data_base)) = view.standard_dims() {
            Self { backing: Backing::Bytes { bytes, inline: false, data_base }, n, pos: 0 }
        } else {
            let count = view.inline_count().unwrap_or(0);
            let data_base = INLINE_HEADER_SIZE;
            Self { backing: Backing::Bytes { bytes, inline: true, data_base }, n: count, pos: 0 }
        }
    }

    /// Read a working arena as an ordered container source (no serialization).
    pub fn from_arena(arena: &'a OpArena) -> Self {
        debug_assert!(arena.is_key_sorted(), "arena source must be key-ascending");
        Self { backing: Backing::Arena(arena), n: arena.container_count(), pos: 0 }
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

    /// Advance past the current container.
    pub fn advance(&mut self) {
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

    /// Advance until the current key ≥ `target`. Returns `true` if it equals it.
    pub fn advance_to(&mut self, target: u16) -> bool {
        while let Some(k) = self.peek_key() {
            if k >= target {
                return k == target;
            }
            self.advance();
        }
        false
    }
}
