//! Zero-copy typed views of a container's payload.
//!
//! A container's bytes are reinterpreted as the natural typed slice for its
//! kind — `&[u16]`, `&[u64; 1024]`, `&[Run]`, or `&[u32]` — via `bytemuck`,
//! relying on the alignment the wire format guarantees (arrays 2-byte, bitmaps
//! 64-byte, runs 2-byte, inline 4-byte). Kernels match on the `Data` enum below
//! instead of decoding bytes by hand.

use bytemuck::{Pod, Zeroable};

use crate::format::*;

/// A bitmap container: 1024 little-endian `u64` words covering a 2^16 key space.
pub type Bitmap = [u64; BITMAP_WORDS];

/// One run of consecutive values, covering `start..=start + len` inclusive.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Run {
    pub start: u16,
    pub len: u16,
}

impl Run {
    /// Inclusive last value of the run.
    #[inline]
    pub fn end(self) -> u16 {
        self.start + self.len
    }
}

/// Typed, zero-copy view of one container's payload. `Inline` appears only for
/// inline-format inputs (a key group of packed `u32`s).
#[derive(Clone, Copy)]
pub enum Data<'a> {
    Array(&'a [u16]),
    Bitmap(&'a Bitmap),
    Run(&'a [Run]),
    Inline(&'a [u32]),
}

impl<'a> Data<'a> {
    /// Reinterpret a container's payload bytes by type.
    pub fn new(typ: u8, card: u32, bytes: &'a [u8]) -> Self {
        match typ {
            CT_ARRAY => Data::Array(bytemuck::cast_slice(&bytes[..card as usize * 2])),
            CT_BITMAP => Data::Bitmap(as_bitmap(&bytes[..BITMAP_BYTES])),
            CT_RUN => {
                let nr = read_u16(bytes, 0) as usize;
                Data::Run(bytemuck::cast_slice(&bytes[2..run_bytes(nr)]))
            }
            CT_INLINE => Data::Inline(bytemuck::cast_slice(&bytes[..card as usize * 4])),
            // The 2-bit type covers exactly these four; corrupt bytes are
            // rejected by `from_bytes` before any kernel builds a `Data`.
            _ => unreachable!("invalid container type {typ}"),
        }
    }

    /// Whether `lo` (a container-local low 16 bits) is present.
    ///
    /// Probes the typed payload directly — one branch on the container form,
    /// then a binary search / bit test over a real slice. Callers that test
    /// *many* values against one container should hoist the match instead (see
    /// `retain_bitmap`), but for a single probe this is the fast path.
    #[inline]
    pub fn contains(&self, lo: u16) -> bool {
        match self {
            Data::Array(a) => a.binary_search(&lo).is_ok(),
            Data::Bitmap(b) => (b[lo as usize / 64] >> (lo % 64)) & 1 == 1,
            Data::Run(runs) => {
                // Runs are sorted by start; the last one starting at or before
                // `lo` is the only one that can contain it.
                let i = runs.partition_point(|r| r.start <= lo);
                i > 0 && lo <= runs[i - 1].end()
            }
            Data::Inline(ids) => ids.binary_search_by(|v| (*v as u16).cmp(&lo)).is_ok(),
        }
    }

    /// Write every low 16 bits, ascending, into `out`; returns the count.
    #[inline]
    pub fn write_sorted(&self, out: &mut [u16]) -> usize {
        if let Data::Array(a) = self {
            out[..a.len()].copy_from_slice(a);
            return a.len();
        }
        let mut k = 0;
        self.for_each(|lo| {
            out[k] = lo;
            k += 1;
        });
        k
    }

    /// Visit every low 16 bits in ascending order.
    #[inline]
    pub fn for_each(&self, mut f: impl FnMut(u16)) {
        match self {
            Data::Array(a) => a.iter().for_each(|&v| f(v)),
            Data::Inline(ids) => ids.iter().for_each(|&v| f(v as u16)),
            Data::Run(runs) => {
                for r in *runs {
                    for v in r.start..=r.end() {
                        f(v);
                    }
                }
            }
            Data::Bitmap(b) => {
                // 8-word groups whose OR is zero skip in one check — sparse
                // bitmaps are mostly empty words, and the flat 1024-word walk
                // was the fixed floor of every low-card extraction.
                for (g, group) in b.chunks_exact(8).enumerate() {
                    if group.iter().fold(0u64, |acc, &w| acc | w) == 0 {
                        continue;
                    }
                    for (w, &word) in group.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            f(((g * 8 + w) * 64) as u16 + bits.trailing_zeros() as u16);
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }
    }
}

#[inline]
pub fn as_bitmap(bytes: &[u8]) -> &Bitmap {
    let words: &[u64] = bytemuck::cast_slice(bytes);
    words.try_into().expect("bitmap payload is BITMAP_WORDS words")
}

#[inline]
pub fn as_bitmap_mut(bytes: &mut [u8]) -> &mut Bitmap {
    let words: &mut [u64] = bytemuck::cast_slice_mut(bytes);
    words.try_into().expect("bitmap payload is BITMAP_WORDS words")
}

