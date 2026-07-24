//! Zero-copy reader over frozen bitmap bytes (e.g. an `mmap`).

use crate::format::*;

/// A frozen bitmap viewed directly from raw bytes — no deserialization.
/// Reads both encodings: standard (`FROZ`) and inline (`FI`).
#[derive(Clone, Copy)]
pub struct FrozenBitmapView<'a> {
    bytes: &'a [u8],
    repr: Repr,
    cardinality: u64,
}

#[derive(Clone, Copy)]
enum Repr {
    Standard { n: usize, data_base: usize },
    Inline { count: usize },
}

impl<'a> FrozenBitmapView<'a> {
    /// Validate and wrap `bytes`. Returns `None` if not a well-formed frozen
    /// bitmap (bad header, out-of-bounds payload, non-ascending keys/values,
    /// or inconsistent cardinality).
    pub fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        if has_inline_magic(bytes) {
            Self::parse_inline(bytes)
        } else {
            Self::parse_standard(bytes)
        }
    }

    fn parse_inline(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < INLINE_HEADER_SIZE {
            return None;
        }
        let count = read_u16(bytes, INLINE_COUNT_OFF) as usize;
        if bytes.len() < INLINE_HEADER_SIZE + count * 4 {
            return None;
        }
        // With values present, they are reinterpreted as `&[u32]` (align 4) when
        // this view feeds an op; reject a base that would fault the zero-copy
        // cast. An empty inline bitmap has nothing to cast, so any base is fine.
        if count > 0 && !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>()) {
            return None;
        }
        let mut prev: Option<u32> = None;
        for i in 0..count {
            let v = read_u32(bytes, INLINE_HEADER_SIZE + i * 4);
            if let Some(p) = prev {
                if v <= p {
                    return None;
                }
            }
            prev = Some(v);
        }
        Some(Self {
            bytes,
            repr: Repr::Inline { count },
            cardinality: count as u64,
        })
    }

    fn parse_standard(bytes: &'a [u8]) -> Option<Self> {
        // Container payloads are reinterpreted as `&[u16]` / `&[Run]` / `&[u64]`
        // (max align 8) zero-copy. Payloads sit at 8- or 64-aligned offsets from
        // the base, so an 8-aligned base makes every one correctly aligned;
        // otherwise the first op would fault inside `bytemuck`.
        if !(bytes.as_ptr() as usize).is_multiple_of(WORD_ALIGN) {
            return None;
        }
        let h = Header::parse(bytes)?;
        let n = h.num_containers as usize;

        if bytes.len() < HEADER_SIZE + index_size(n) {
            return None;
        }
        let data_base = data_section_off(n, h.has_bitmap);
        if bytes.len() < data_base {
            return None;
        }

        let mut prev_key: Option<u16> = None;
        let mut sum_card: u64 = 0;
        for i in 0..n {
            let e = read_index_entry(bytes, n, i);
            match prev_key {
                Some(pk) if e.key <= pk => return None,
                _ => prev_key = Some(e.key),
            }
            sum_card += e.cardinality as u64;

            let start = data_base + e.data_offset as usize;
            let size = match e.typ {
                CT_ARRAY => e.cardinality as usize * 2,
                CT_BITMAP => BITMAP_BYTES,
                CT_RUN => {
                    if start + 2 > bytes.len() {
                        return None;
                    }
                    2 + read_u16(bytes, start) as usize * 4
                }
                _ => return None,
            };
            if start.checked_add(size)? > bytes.len() {
                return None;
            }
            // Payload bytes are in-bounds; now validate their *content*, so the
            // structural guarantees every reader and kernel relies on actually
            // hold — sorted-unique arrays, in-range non-overlapping runs, and
            // cardinalities that match the payload. Without this the type is
            // "structurally valid" only, and corrupt-but-well-sized bytes turn
            // into wrong answers, panics, or (in the merge kernels) OOB writes.
            if !payload_is_valid(bytes, e.typ, e.cardinality, start) {
                return None;
            }
        }
        if sum_card != h.cardinality {
            return None;
        }

        Some(Self {
            bytes,
            repr: Repr::Standard { n, data_base },
            cardinality: h.cardinality,
        })
    }

    /// Skip validation; caller guarantees frostbit-produced bytes.
    pub(crate) fn from_bytes_trusted(bytes: &'a [u8]) -> Self {
        if has_inline_magic(bytes) {
            let count = read_u16(bytes, INLINE_COUNT_OFF) as usize;
            return Self {
                bytes,
                repr: Repr::Inline { count },
                cardinality: count as u64,
            };
        }
        let h = Header::parse(bytes).expect("trusted bytes must carry a valid header");
        let n = h.num_containers as usize;
        Self {
            bytes,
            repr: Repr::Standard {
                n,
                data_base: data_section_off(n, h.has_bitmap),
            },
            cardinality: h.cardinality,
        }
    }

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Whether this view reads the inline (`FI`) encoding.
    #[inline]
    pub fn is_inline(&self) -> bool {
        matches!(self.repr, Repr::Inline { .. })
    }

    /// Total set bits.
    #[inline]
    pub fn len(&self) -> u64 {
        self.cardinality
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Containers in the standard index; `0` for inline (no index).
    #[inline]
    pub fn num_containers(&self) -> usize {
        match self.repr {
            Repr::Standard { n, .. } => n,
            Repr::Inline { .. } => 0,
        }
    }

    /// Standard-format `(num_containers, data_base)`, or `None` if inline.
    #[inline]
    pub(crate) fn standard_dims(&self) -> Option<(usize, usize)> {
        match self.repr {
            Repr::Standard { n, data_base } => Some((n, data_base)),
            Repr::Inline { .. } => None,
        }
    }

    /// Inline value count, or `None` if standard.
    #[inline]
    pub(crate) fn inline_count(&self) -> Option<usize> {
        match self.repr {
            Repr::Inline { count } => Some(count),
            Repr::Standard { .. } => None,
        }
    }

    /// Iterate all values in ascending order.
    pub fn iter(&self) -> Iter<'a> {
        let mode = match self.repr {
            Repr::Inline { count } => Mode::Inline { pos: 0, count },
            Repr::Standard { n, data_base } => Mode::Standard {
                n,
                data_base,
                ci: 0,
                hi: 0,
                cont: Cont::Done,
            },
        };
        Iter {
            bytes: self.bytes,
            mode,
            remaining: self.cardinality,
        }
    }

    /// Whether `value` is set. O(log containers) + per-container probe.
    pub fn contains(&self, value: u32) -> bool {
        match self.repr {
            Repr::Inline { count } => inline_contains(self.bytes, count, value),
            Repr::Standard { n, data_base } => {
                let Some(i) = find_container(self.bytes, n, (value >> 16) as u16) else {
                    return false;
                };
                let e = read_index_entry(self.bytes, n, i);
                let lo = value as u16;
                let start = data_base + e.data_offset as usize;
                match e.typ {
                    CT_ARRAY => array_contains(self.bytes, start, e.cardinality as usize, lo),
                    CT_BITMAP => {
                        let w = read_u64(self.bytes, start + (lo as usize / 64) * 8);
                        (w >> (lo as usize % 64)) & 1 == 1
                    }
                    CT_RUN => run_contains(self.bytes, start, lo),
                    _ => false,
                }
            }
        }
    }

    /// Smallest value, or `None` if empty.
    pub fn min(&self) -> Option<u32> {
        match self.repr {
            Repr::Inline { count } => {
                (count > 0).then(|| read_u32(self.bytes, INLINE_HEADER_SIZE))
            }
            Repr::Standard { n, .. } => {
                if n == 0 {
                    return None;
                }
                let e = read_index_entry(self.bytes, n, 0);
                Some(((e.key as u32) << 16) | self.container_min(&e) as u32)
            }
        }
    }

    /// Largest value, or `None` if empty.
    pub fn max(&self) -> Option<u32> {
        match self.repr {
            Repr::Inline { count } => {
                (count > 0).then(|| read_u32(self.bytes, INLINE_HEADER_SIZE + (count - 1) * 4))
            }
            Repr::Standard { n, .. } => {
                if n == 0 {
                    return None;
                }
                let e = read_index_entry(self.bytes, n, n - 1);
                Some(((e.key as u32) << 16) | self.container_max(&e) as u32)
            }
        }
    }

    #[inline]
    fn data_start(&self, e: &IndexEntry) -> usize {
        match self.repr {
            Repr::Standard { data_base, .. } => data_base + e.data_offset as usize,
            Repr::Inline { .. } => unreachable!("no containers in inline form"),
        }
    }

    fn container_min(&self, e: &IndexEntry) -> u16 {
        let start = self.data_start(e);
        match e.typ {
            CT_ARRAY => read_u16(self.bytes, start),
            CT_RUN => read_u16(self.bytes, start + 2),
            CT_BITMAP => self.bitmap_first(start),
            _ => 0,
        }
    }

    fn container_max(&self, e: &IndexEntry) -> u16 {
        let start = self.data_start(e);
        match e.typ {
            CT_ARRAY => read_u16(self.bytes, start + (e.cardinality as usize - 1) * 2),
            CT_RUN => {
                let nr = read_u16(self.bytes, start) as usize;
                let off = start + 2 + (nr - 1) * 4;
                read_u16(self.bytes, off) + read_u16(self.bytes, off + 2)
            }
            CT_BITMAP => self.bitmap_last(start),
            _ => 0,
        }
    }

    fn bitmap_first(&self, start: usize) -> u16 {
        for w in 0..BITMAP_WORDS {
            let word = read_u64(self.bytes, start + w * 8);
            if word != 0 {
                return (w * 64 + word.trailing_zeros() as usize) as u16;
            }
        }
        0
    }

    fn bitmap_last(&self, start: usize) -> u16 {
        for w in (0..BITMAP_WORDS).rev() {
            let word = read_u64(self.bytes, start + w * 8);
            if word != 0 {
                return (w * 64 + 63 - word.leading_zeros() as usize) as u16;
            }
        }
        0
    }
}

/// Ascending iterator over a frozen bitmap's values.
pub struct Iter<'a> {
    bytes: &'a [u8],
    mode: Mode,
    remaining: u64,
}

enum Mode {
    Inline { pos: usize, count: usize },
    Standard { n: usize, data_base: usize, ci: usize, hi: u32, cont: Cont },
}

enum Cont {
    Done,
    Array { start: usize, card: usize, i: usize },
    Bitmap { start: usize, word_idx: usize, word: u64 },
    // Run cursor in u32 space so an end of 0xFFFF can't wrap.
    Run { start: usize, nr: usize, ri: usize, cur: u32, end: u32 },
}

impl Cont {
    fn next_lo(&mut self, bytes: &[u8]) -> Option<u32> {
        match self {
            Self::Done => None,
            Self::Array { start, card, i } => {
                if *i >= *card {
                    return None;
                }
                let v = read_u16(bytes, *start + *i * 2) as u32;
                *i += 1;
                Some(v)
            }
            Self::Bitmap { start, word_idx, word } => loop {
                if *word != 0 {
                    let tz = word.trailing_zeros();
                    *word &= *word - 1;
                    return Some((*word_idx as u32) * 64 + tz);
                }
                *word_idx += 1;
                if *word_idx >= BITMAP_WORDS {
                    return None;
                }
                *word = read_u64(bytes, *start + *word_idx * 8);
            },
            Self::Run { start, nr, ri, cur, end } => {
                if *cur > *end {
                    *ri += 1;
                    if *ri >= *nr {
                        return None;
                    }
                    let off = *start + 2 + *ri * 4;
                    *cur = read_u16(bytes, off) as u32;
                    *end = *cur + read_u16(bytes, off + 2) as u32;
                }
                let v = *cur;
                *cur += 1;
                Some(v)
            }
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        match &mut self.mode {
            Mode::Inline { pos, count } => {
                if *pos >= *count {
                    return None;
                }
                let v = read_u32(self.bytes, INLINE_HEADER_SIZE + *pos * 4);
                *pos += 1;
                self.remaining -= 1;
                Some(v)
            }
            Mode::Standard { n, data_base, ci, hi, cont } => loop {
                if let Some(lo) = cont.next_lo(self.bytes) {
                    self.remaining -= 1;
                    return Some(*hi | lo);
                }
                if *ci >= *n {
                    return None;
                }
                let e = read_index_entry(self.bytes, *n, *ci);
                *ci += 1;
                *hi = (e.key as u32) << 16;
                let start = *data_base + e.data_offset as usize;
                *cont = match e.typ {
                    CT_ARRAY => Cont::Array { start, card: e.cardinality as usize, i: 0 },
                    CT_BITMAP => Cont::Bitmap {
                        start,
                        word_idx: 0,
                        word: read_u64(self.bytes, start),
                    },
                    _ => {
                        let nr = read_u16(self.bytes, start) as usize;
                        let cur = read_u16(self.bytes, start + 2) as u32;
                        let end = cur + read_u16(self.bytes, start + 4) as u32;
                        Cont::Run { start, nr, ri: 0, cur, end }
                    }
                };
            },
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let lo = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        (lo, usize::try_from(self.remaining).ok())
    }
}

impl std::iter::FusedIterator for Iter<'_> {}

impl<'a> IntoIterator for FrozenBitmapView<'a> {
    type Item = u32;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl<'a> IntoIterator for &FrozenBitmapView<'a> {
    type Item = u32;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl PartialEq for FrozenBitmapView<'_> {
    /// Set equality: equal iff they hold the same values, regardless of encoding
    /// (inline vs standard) or backing bytes.
    fn eq(&self, other: &Self) -> bool {
        self.cardinality == other.cardinality && self.iter().eq(other.iter())
    }
}
impl Eq for FrozenBitmapView<'_> {}

fn inline_contains(bytes: &[u8], count: usize, value: u32) -> bool {
    let (mut lo, mut hi) = (0usize, count);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match read_u32(bytes, INLINE_HEADER_SIZE + mid * 4).cmp(&value) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn array_contains(bytes: &[u8], start: usize, card: usize, lo16: u16) -> bool {
    let (mut lo, mut hi) = (0usize, card);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match read_u16(bytes, start + mid * 2).cmp(&lo16) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn run_contains(bytes: &[u8], start: usize, lo16: u16) -> bool {
    let nr = read_u16(bytes, start) as usize;
    let v = lo16 as u32;
    let (mut lo, mut hi) = (0usize, nr);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let s = read_u16(bytes, start + 2 + mid * 4) as u32;
        let e = s + read_u16(bytes, start + 2 + mid * 4 + 2) as u32;
        if v < s {
            hi = mid;
        } else if v > e {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// Validate one container's payload *content*, given its bytes are already known
/// to be in-bounds. Enforces the invariants the readers and kernels assume:
/// - **array**: lows strictly ascending (⇒ sorted and unique);
/// - **run**: at least one run, each `start + len ≤ 0xFFFF` (no `u16` wrap),
///   runs ascending and non-overlapping, `Σ(len + 1)` equal to `card`;
/// - **bitmap**: exactly `card` bits set.
///
/// Only called from [`FrozenBitmapView::from_bytes`] (the untrusted boundary);
/// the trusted path skips it, so frostbit-produced bitmaps pay nothing.
fn payload_is_valid(bytes: &[u8], typ: u8, card: u32, start: usize) -> bool {
    match typ {
        CT_ARRAY => {
            let mut prev: Option<u16> = None;
            for j in 0..card as usize {
                let v = read_u16(bytes, start + j * 2);
                if prev.is_some_and(|p| v <= p) {
                    return false;
                }
                prev = Some(v);
            }
            true
        }
        CT_BITMAP => {
            let mut pop = 0u32;
            for w in 0..BITMAP_WORDS {
                pop += read_u64(bytes, start + w * 8).count_ones();
            }
            pop == card
        }
        CT_RUN => {
            let nr = read_u16(bytes, start) as usize;
            // A canonical run container has 1..=MAX_RUNS runs (more would have
            // been stored as a bitmap). Rejecting the rest also keeps the
            // planner's "slot capacity ≤ BITMAP_BYTES" invariant intact (SAFE-11).
            if nr == 0 || nr > MAX_RUNS {
                return false;
            }
            let (mut total, mut prev_end): (u32, Option<u32>) = (0, None);
            for r in 0..nr {
                let off = start + 2 + r * 4;
                let s = read_u16(bytes, off) as u32;
                let len = read_u16(bytes, off + 2) as u32;
                let end = s + len;
                // In range (no u16 wrap) and strictly past the previous run's
                // end (ascending, non-overlapping — adjacency is allowed).
                if end > 0xFFFF || prev_end.is_some_and(|pe| s <= pe) {
                    return false;
                }
                total += len + 1;
                prev_end = Some(end);
            }
            total == card
        }
        _ => false,
    }
}
