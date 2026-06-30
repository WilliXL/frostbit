//! Ascending per-container cursor over a view, for both encodings. Shared by
//! the analysis pass and (later) the op kernels — mirrors the monorepo
//! `ContainerIter`.

use crate::container::Data;
use crate::format::*;
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

pub struct ContainerCursor<'a> {
    bytes: &'a [u8],
    inline: bool,
    // Standard: container index. Inline: packed-u32 region.
    n: usize,
    data_base: usize,
    // Cursor position: container index (standard) or value index (inline).
    pos: usize,
}

impl<'a> ContainerCursor<'a> {
    pub fn new(view: &FrozenBitmapView<'a>) -> Self {
        let bytes = view.as_bytes();
        if let Some((n, data_base)) = view.standard_dims() {
            Self { bytes, inline: false, n, data_base, pos: 0 }
        } else {
            let count = view.inline_count().unwrap_or(0);
            Self { bytes, inline: true, n: count, data_base: INLINE_HEADER_SIZE, pos: 0 }
        }
    }

    /// Key of the current container, or `None` when exhausted.
    #[inline]
    pub fn peek_key(&self) -> Option<u16> {
        if self.pos >= self.n {
            return None;
        }
        Some(if self.inline {
            (read_u32(self.bytes, self.data_base + self.pos * 4) >> 16) as u16
        } else {
            read_key(self.bytes, self.n, self.pos)
        })
    }

    /// Current container without advancing.
    pub fn get(&self) -> ContainerRef<'a> {
        debug_assert!(self.pos < self.n);
        if self.inline {
            let key = (read_u32(self.bytes, self.data_base + self.pos * 4) >> 16) as u16;
            let mut end = self.pos + 1;
            while end < self.n
                && (read_u32(self.bytes, self.data_base + end * 4) >> 16) as u16 == key
            {
                end += 1;
            }
            ContainerRef {
                key,
                typ: CT_INLINE,
                card: (end - self.pos) as u32,
                data: &self.bytes[self.data_base + self.pos * 4..self.data_base + end * 4],
            }
        } else {
            let e = read_index_entry(self.bytes, self.n, self.pos);
            let start = self.data_base + e.data_offset as usize;
            let size = match e.typ {
                CT_ARRAY => e.cardinality as usize * 2,
                CT_BITMAP => BITMAP_BYTES,
                CT_RUN => 2 + read_u16(self.bytes, start) as usize * 4,
                _ => 0,
            };
            ContainerRef {
                key: e.key,
                typ: e.typ,
                card: e.cardinality,
                data: &self.bytes[start..start + size],
            }
        }
    }

    /// Advance past the current container.
    pub fn advance(&mut self) {
        if self.inline {
            let Some(key) = self.peek_key() else { return };
            while self.pos < self.n
                && (read_u32(self.bytes, self.data_base + self.pos * 4) >> 16) as u16 == key
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
