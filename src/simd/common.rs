//! Primitives shared by more than one SIMD op.
//!
//! The window scan (is `v` in `freq[f..f + W]`?), the branchless lane-compaction
//! table, the NEON word-fold skeletons, and the whole-bitmap build/count
//! helpers. Anything used by a single op lives with that op instead.

use crate::container::{Bitmap, Run};
use crate::format::BITMAP_WORDS;

// --- block-size heuristic ------------------------------------------------------
/// At or below this size ratio the arrays are balanced enough that the
/// shuffle-merge's 8-at-a-time compare wins (one horizontal reduction per block,
/// not per element). Above it, one side is "rare" — a scan of the larger wins.
pub(super) const MERGE_MAX_RATIO: usize = 4;

// --- lane compaction table -----------------------------------------------------
/// Byte-shuffle indices that compact the set lanes of a `u16x8` to the front,
/// keyed by the 8-bit lane mask. Unused trailing bytes are `0xFF` (≥16), which
/// both NEON `vqtbl1q` and x86 `pshufb` map to zero. `popcount(mask)` lanes valid.
pub(super) const COMPACT: [[u8; 16]; 256] = {
    let mut t = [[0xFFu8; 16]; 256];
    let mut m = 0usize;
    while m < 256 {
        let (mut pos, mut lane) = (0usize, 0usize);
        while lane < 8 {
            if m & (1 << lane) != 0 {
                t[m][2 * pos] = (2 * lane) as u8;
                t[m][2 * pos + 1] = (2 * lane + 1) as u8;
                pos += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    t
};

// --- window scan ---------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn window_has_neon(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::aarch64::*;
    let win = vld1q_u16(freq.as_ptr().add(f));
    vmaxvq_u16(vceqq_u16(win, vdupq_n_u16(v))) != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(super) unsafe fn window_has_sse2(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::x86_64::*;
    let win = _mm_loadu_si128(freq.as_ptr().add(f).cast());
    _mm_movemask_epi8(_mm_cmpeq_epi16(win, _mm_set1_epi16(v as i16))) != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn window_has_avx2(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::x86_64::*;
    let win = _mm256_loadu_si256(freq.as_ptr().add(f).cast());
    _mm256_movemask_epi8(_mm256_cmpeq_epi16(win, _mm256_set1_epi16(v as i16))) != 0
}
// --- whole-bitmap build / clear / scatter --------------------------------------
#[inline]
pub fn copy(dst: &mut Bitmap, src: &Bitmap) {
    dst.copy_from_slice(src);
}

#[inline]
pub fn clear(dst: &mut Bitmap) {
    dst.fill(0);
}

#[inline]
pub fn set_values(dst: &mut Bitmap, vals: &[u16]) {
    for &v in vals {
        dst[v as usize / 64] |= 1u64 << (v as usize % 64);
    }
}

#[inline]
pub fn clear_values(dst: &mut Bitmap, vals: &[u16]) {
    for &v in vals {
        dst[v as usize / 64] &= !(1u64 << (v as usize % 64));
    }
}

pub fn set_runs(dst: &mut Bitmap, runs: &[Run]) {
    for r in runs {
        fill(dst, r.start as usize, r.end() as usize, true);
    }
}

pub fn clear_runs(dst: &mut Bitmap, runs: &[Run]) {
    for r in runs {
        fill(dst, r.start as usize, r.end() as usize, false);
    }
}

/// Set (or clear) the inclusive bit range `[lo, hi]` via word masks.
#[inline]
fn fill(dst: &mut Bitmap, lo: usize, hi: usize, set: bool) {
    let (wl, wh) = (lo / 64, hi / 64);
    let lo_mask = u64::MAX << (lo % 64);
    let hi_mask = u64::MAX >> (63 - hi % 64);
    if wl == wh {
        let m = lo_mask & hi_mask;
        if set {
            dst[wl] |= m;
        } else {
            dst[wl] &= !m;
        }
    } else {
        let full = if set { u64::MAX } else { 0 };
        if set {
            dst[wl] |= lo_mask;
            dst[wh] |= hi_mask;
        } else {
            dst[wl] &= !lo_mask;
            dst[wh] &= !hi_mask;
        }
        dst[wl + 1..wh].fill(full);
    }
}
// --- population count ----------------------------------------------------------
#[inline]
pub fn popcount(b: &Bitmap) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            return popcount_avx512(b);
        }
        return popcount_scalar(b);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return popcount_neon(b);
    }
    #[allow(unreachable_code)]
    popcount_scalar(b)
}

fn popcount_scalar(b: &Bitmap) -> u32 {
    b.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn popcount_neon(b: &Bitmap) -> u32 {
    use std::arch::aarch64::*;
    let p = b.as_ptr().cast::<u8>();
    let mut acc = vdupq_n_u16(0);
    for i in (0..BITMAP_WORDS * 8).step_by(16) {
        acc = vpadalq_u8(acc, vcntq_u8(vld1q_u8(p.add(i))));
    }
    vaddlvq_u16(acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn popcount_avx512(b: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let p = b.as_ptr().cast::<__m512i>();
    let mut acc = _mm512_setzero_si512();
    for i in 0..BITMAP_WORDS / 8 {
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_loadu_si512(p.add(i))));
    }
    _mm512_reduce_add_epi64(acc) as u32
}
// --- shared aarch64 NEON word loops -----------------------------------------
//
// NEON is baseline on aarch64, so its intrinsics can be called from these
// generic helpers (the `combine` closure is monomorphized inline). x86 cannot
// do this for non-baseline features, so its loops are written out per op.

/// `dst = combine(dst, src)` word-by-word (two `u64` lanes per step).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn fold_neon(
    dst: &mut Bitmap,
    src: &Bitmap,
    combine: impl Fn(std::arch::aarch64::uint64x2_t, std::arch::aarch64::uint64x2_t) -> std::arch::aarch64::uint64x2_t,
) {
    use std::arch::aarch64::*;
    for i in (0..BITMAP_WORDS).step_by(2) {
        let r = combine(vld1q_u64(dst.as_ptr().add(i)), vld1q_u64(src.as_ptr().add(i)));
        vst1q_u64(dst.as_mut_ptr().add(i), r);
    }
}

/// `dst = combine(dst, src)` with a fused population count of the result. Per-
/// byte `CNT` accumulated into `u16` lanes (`vpadalq_u8`), reduced once.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(super) unsafe fn fold_count_neon(
    dst: &mut Bitmap,
    src: &Bitmap,
    combine: impl Fn(std::arch::aarch64::uint64x2_t, std::arch::aarch64::uint64x2_t) -> std::arch::aarch64::uint64x2_t,
) -> u32 {
    use std::arch::aarch64::*;
    // Four independent count accumulators — a single `vpadalq` chain is
    // latency-bound well below the streaming rate (see bitmap_andnot).
    let (mut c0, mut c1, mut c2, mut c3) =
        (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
    for i in (0..BITMAP_WORDS).step_by(8) {
        let r0 = combine(vld1q_u64(dst.as_ptr().add(i)), vld1q_u64(src.as_ptr().add(i)));
        let r1 = combine(vld1q_u64(dst.as_ptr().add(i + 2)), vld1q_u64(src.as_ptr().add(i + 2)));
        let r2 = combine(vld1q_u64(dst.as_ptr().add(i + 4)), vld1q_u64(src.as_ptr().add(i + 4)));
        let r3 = combine(vld1q_u64(dst.as_ptr().add(i + 6)), vld1q_u64(src.as_ptr().add(i + 6)));
        vst1q_u64(dst.as_mut_ptr().add(i), r0);
        vst1q_u64(dst.as_mut_ptr().add(i + 2), r1);
        vst1q_u64(dst.as_mut_ptr().add(i + 4), r2);
        vst1q_u64(dst.as_mut_ptr().add(i + 6), r3);
        c0 = vpadalq_u8(c0, vcntq_u8(vreinterpretq_u8_u64(r0)));
        c1 = vpadalq_u8(c1, vcntq_u8(vreinterpretq_u8_u64(r1)));
        c2 = vpadalq_u8(c2, vcntq_u8(vreinterpretq_u8_u64(r2)));
        c3 = vpadalq_u8(c3, vcntq_u8(vreinterpretq_u8_u64(r3)));
    }
    vaddlvq_u16(vaddq_u16(vaddq_u16(c0, c1), vaddq_u16(c2, c3)))
}
