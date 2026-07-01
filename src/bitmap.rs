//! Owned frozen bitmap: a 64-byte-aligned byte buffer in frozen wire format.

use std::fmt;
use std::ops::Deref;

use aligned_vec::{AVec, ConstAlign};

use crate::format::BUF_ALIGN;

/// 64-byte-aligned owned byte buffer backing a [`FrozenBitmap`].
pub(crate) type AlignedBuf = AVec<u8, ConstAlign<BUF_ALIGN>>;

/// Allocate an empty [`AlignedBuf`] with `cap` bytes reserved.
pub(crate) fn aligned_buf(cap: usize) -> AlignedBuf {
    AVec::with_capacity(BUF_ALIGN, cap)
}

/// Per-thread pool of result buffers. A [`FrozenBitmap`] takes its backing
/// buffer from here on construction and returns it on drop, so a repeated op
/// (`materialize` / `*_fast` in a loop) reuses one aligned allocation — no malloc
/// on the hot path once warm, matching the arena/cursor/stack scratch pools.
mod result_pool {
    use std::cell::RefCell;

    use super::{aligned_buf, AlignedBuf};

    const MAX_POOLED: usize = 8;
    thread_local! {
        static POOL: RefCell<Vec<AlignedBuf>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn take(cap: usize) -> AlignedBuf {
        match POOL.with(|p| p.borrow_mut().pop()) {
            Some(mut buf) => {
                buf.clear();
                buf.reserve(cap);
                buf
            }
            None => aligned_buf(cap),
        }
    }

    pub(super) fn put(buf: AlignedBuf) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < MAX_POOLED {
                p.push(buf);
            }
        });
    }
}

/// A result buffer from the per-thread pool, with `cap` bytes reserved.
pub(crate) fn result_buf(cap: usize) -> AlignedBuf {
    result_pool::take(cap)
}

/// An owned frozen bitmap. The backing allocation is 64-byte aligned so bitmap
/// container payloads sit on cache-line boundaries for SIMD. Produced by the
/// builder, set ops, and roaring conversion; query it via [`Self::as_bytes`]
/// (a `FrozenBitmapView` reader lands in a later step).
pub struct FrozenBitmap {
    buf: AlignedBuf,
}

impl Clone for FrozenBitmap {
    fn clone(&self) -> Self {
        let mut buf = result_buf(self.buf.len());
        buf.extend_from_slice(&self.buf);
        Self { buf }
    }
}

impl Drop for FrozenBitmap {
    fn drop(&mut self) {
        // Return the backing buffer to the per-thread pool for the next op.
        result_pool::put(std::mem::replace(&mut self.buf, aligned_buf(0)));
    }
}

impl FrozenBitmap {
    /// Validate and copy frozen-bitmap `bytes` into a 64-byte-aligned buffer.
    /// `None` if `bytes` is not a well-formed frozen bitmap.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        crate::FrozenBitmapView::from_bytes(bytes)?;
        let mut buf = result_buf(bytes.len());
        buf.extend_from_slice(bytes);
        Some(Self { buf })
    }

    pub(crate) fn from_buf(buf: AlignedBuf) -> Self {
        Self { buf }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Zero-copy view. O(1): bytes are valid by construction.
    #[inline]
    pub fn view(&self) -> crate::FrozenBitmapView<'_> {
        crate::FrozenBitmapView::from_bytes_trusted(&self.buf)
    }

    /// Iterate all values in ascending order.
    #[inline]
    pub fn iter(&self) -> crate::Iter<'_> {
        self.view().iter()
    }

    /// Serialized size in bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.buf.len()
    }
}

impl Deref for FrozenBitmap {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl AsRef<[u8]> for FrozenBitmap {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.buf
    }
}

impl PartialEq for FrozenBitmap {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.buf[..] == other.buf[..]
    }
}
impl Eq for FrozenBitmap {}

impl fmt::Debug for FrozenBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrozenBitmap")
            .field("bytes", &self.byte_len())
            .finish()
    }
}
