//! Owned frozen bitmap: a 64-byte-aligned byte buffer in frozen wire format.

use std::fmt;

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
/// builder, set ops, and roaring conversion. Query it directly
/// ([`contains`](Self::contains), [`len`](Self::len), [`min`](Self::min) /
/// [`max`](Self::max), [`iter`](Self::iter)) or take a zero-copy
/// [`view`](Self::view); the raw bytes are [`as_bytes`](Self::as_bytes).
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
    /// The empty frozen bitmap.
    pub fn empty() -> Self {
        crate::FrozenBitmapBuilder::new().finish()
    }

    /// Validate and copy frozen-bitmap `bytes` into a 64-byte-aligned buffer.
    /// `None` if `bytes` is not a well-formed frozen bitmap.
    ///
    /// Any source alignment is accepted: the bytes are copied into the aligned
    /// owned buffer first and that (always op-safe) copy is validated, so unlike
    /// [`FrozenBitmapView::from_bytes`](crate::FrozenBitmapView::from_bytes)
    /// there is no base-alignment precondition.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut buf = result_buf(bytes.len());
        buf.extend_from_slice(bytes);
        crate::FrozenBitmapView::from_bytes(&buf)?;
        Some(Self { buf })
    }

    pub(crate) fn from_buf(buf: AlignedBuf) -> Self {
        Self { buf }
    }

    /// Copy already-valid frozen bytes into an owned, aligned buffer, skipping
    /// validation. The caller guarantees `bytes` came from a valid frozen source
    /// (e.g. a live [`FrozenBitmapView`], whose bytes were validated when it was
    /// parsed) — used on the hot materialize path where re-validating a trusted
    /// leaf would be pure overhead.
    pub(crate) fn from_bytes_trusted(bytes: &[u8]) -> Self {
        let mut buf = result_buf(bytes.len());
        buf.extend_from_slice(bytes);
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

    /// Serialized size in bytes (not the cardinality — see [`len`](Self::len)).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.buf.len()
    }

    /// Number of values in the set (cardinality).
    #[inline]
    pub fn len(&self) -> u64 {
        self.view().len()
    }

    /// Whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.view().is_empty()
    }

    /// Whether `value` is in the set. O(log containers) + a per-container probe.
    #[inline]
    pub fn contains(&self, value: u32) -> bool {
        self.view().contains(value)
    }

    /// Smallest value, or `None` if the set is empty.
    #[inline]
    pub fn min(&self) -> Option<u32> {
        self.view().min()
    }

    /// Largest value, or `None` if the set is empty.
    #[inline]
    pub fn max(&self) -> Option<u32> {
        self.view().max()
    }
}

impl AsRef<[u8]> for FrozenBitmap {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.buf
    }
}

impl<'a> IntoIterator for &'a FrozenBitmap {
    type Item = u32;
    type IntoIter = crate::Iter<'a>;
    #[inline]
    fn into_iter(self) -> crate::Iter<'a> {
        self.iter()
    }
}

impl PartialEq for FrozenBitmap {
    /// Set equality (same values), with a fast path for byte-identical buffers.
    /// Two set-equal bitmaps in different encodings (e.g. inline vs standard)
    /// compare equal.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.buf[..] == other.buf[..] || self.view() == other.view()
    }
}
impl Eq for FrozenBitmap {}

impl std::hash::Hash for FrozenBitmap {
    /// Hashes the value set, consistent with the set-equality [`PartialEq`].
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let v = self.view();
        v.len().hash(state);
        for value in v.iter() {
            value.hash(state);
        }
    }
}

impl fmt::Debug for FrozenBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrozenBitmap")
            .field("bytes", &self.byte_len())
            .finish()
    }
}
