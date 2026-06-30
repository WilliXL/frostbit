//! Dense container primitives over typed slices.
//!
//! Every function here is a straight loop over a fixed-size `[u64; 1024]`
//! bitmap or a sorted `&[u16]`. At `opt-level >= 2` LLVM autovectorizes these
//! to the target's SIMD (NEON / AVX), so the hot kernels stay branch-free and
//! readable with no `unsafe`. If a profile ever shows a specific gap, add an
//! explicit `std::arch` path *behind the same function name* — callers never
//! change.

use crate::container::{Bitmap, Run};
use crate::format::BITMAP_WORDS;

// --- bitmap word ops --------------------------------------------------------

#[inline]
pub fn and(dst: &mut Bitmap, src: &Bitmap) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= *s;
    }
}

#[inline]
pub fn or(dst: &mut Bitmap, src: &Bitmap) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

#[inline]
pub fn andnot(dst: &mut Bitmap, src: &Bitmap) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= !*s;
    }
}

#[inline]
pub fn copy(dst: &mut Bitmap, src: &Bitmap) {
    dst.copy_from_slice(src);
}

#[inline]
pub fn clear(dst: &mut Bitmap) {
    dst.fill(0);
}

#[inline]
pub fn popcount(b: &Bitmap) -> u32 {
    // x86_64 has a scalar POPCNT instruction, so the loop is already optimal.
    // aarch64 has no scalar popcount, so `count_ones` in a loop is slow —
    // use the NEON `CNT` (per-byte popcount) with a widening reduction.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { popcount_neon(b) };
    }
    #[allow(unreachable_code)]
    b.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn popcount_neon(b: &Bitmap) -> u32 {
    use std::arch::aarch64::*;
    let p = b.as_ptr() as *const u8;
    let mut acc = vdupq_n_u16(0);
    // 16 bytes per step: per-byte CNT accumulated into u16 lanes, reduced once.
    for i in (0..BITMAP_BYTES).step_by(16) {
        acc = vpadalq_u8(acc, vcntq_u8(vld1q_u8(p.add(i))));
    }
    vaddlvq_u16(acc)
}

const BITMAP_BYTES: usize = BITMAP_WORDS * 8;

// --- fused op + popcount (one pass over the bitmap) -------------------------

/// `dst &= src`, returning the result's population count in a single pass.
#[inline]
pub fn and_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { fold_count_neon(dst, src, |a, b| vandq_u64(a, b)) };
    }
    #[allow(unreachable_code)]
    {
        let mut c = 0;
        for (d, s) in dst.iter_mut().zip(src) {
            *d &= *s;
            c += d.count_ones();
        }
        c
    }
}

/// `dst |= src`, returning the result's population count in a single pass.
#[inline]
pub fn or_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        return unsafe { fold_count_neon(dst, src, |a, b| vorrq_u64(a, b)) };
    }
    #[allow(unreachable_code)]
    {
        let mut c = 0;
        for (d, s) in dst.iter_mut().zip(src) {
            *d |= *s;
            c += d.count_ones();
        }
        c
    }
}

/// `dst &= !src`, returning the result's population count in a single pass.
#[inline]
pub fn andnot_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // BIC computes `a & !b`.
        return unsafe { fold_count_neon(dst, src, |a, b| vbicq_u64(a, b)) };
    }
    #[allow(unreachable_code)]
    {
        let mut c = 0;
        for (d, s) in dst.iter_mut().zip(src) {
            *d &= !*s;
            c += d.count_ones();
        }
        c
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fold_count_neon(
    dst: &mut Bitmap,
    src: &Bitmap,
    combine: impl Fn(std::arch::aarch64::uint64x2_t, std::arch::aarch64::uint64x2_t) -> std::arch::aarch64::uint64x2_t,
) -> u32 {
    use std::arch::aarch64::*;
    // Accumulate per-byte popcounts into u16 lanes (`vpadalq_u8`) and reduce
    // once at the end; the per-iteration cost is just CNT + pairwise-accumulate.
    let mut acc = vdupq_n_u16(0);
    for i in (0..BITMAP_WORDS).step_by(2) {
        let r = combine(vld1q_u64(dst.as_ptr().add(i)), vld1q_u64(src.as_ptr().add(i)));
        vst1q_u64(dst.as_mut_ptr().add(i), r);
        acc = vpadalq_u8(acc, vcntq_u8(vreinterpretq_u8_u64(r)));
    }
    vaddlvq_u16(acc)
}

// --- bitmap point/run population --------------------------------------------

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

/// Extract set bits into `out` in ascending order; returns the count. Used to
/// downgrade a bitmap accumulator to an array.
pub fn to_array(b: &Bitmap, out: &mut [u16]) -> usize {
    let mut k = 0;
    for (w, &word) in b.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            out[k] = (w * 64) as u16 + bits.trailing_zeros() as u16;
            k += 1;
            bits &= bits - 1;
        }
    }
    k
}

const _: () = assert!(BITMAP_WORDS == 1024);

// --- sorted u16 array ops (two-pointer) -------------------------------------

/// `a ∩ b` for sorted, unique slices. Returns the result length.
///
/// Dispatches to a SIMD "broadcast scan" (NEON / SSE2): for each value of the
/// smaller side, gallop a window of the larger side forward and test all lanes
/// at once. Falls back to a scalar two-pointer merge elsewhere. `out` must not
/// alias either input.
#[inline]
pub fn array_intersect(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { intersect_neon(a, b, out) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        return unsafe { intersect_sse2(a, b, out) };
    }
    #[allow(unreachable_code)]
    intersect_scalar(a, b, out)
}

/// Scalar two-pointer reference (and fallback for non-SIMD targets).
#[inline]
fn intersect_scalar(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out[k] = a[i];
                k += 1;
                i += 1;
                j += 1;
            }
        }
    }
    k
}

/// Broadcast-scan body shared by the SIMD paths: `window8(freq, f)` returns
/// whether `v` is in the 8-lane window `freq[f..f + 8]`.
#[inline(always)]
fn broadcast_scan(
    a: &[u16],
    b: &[u16],
    out: &mut [u16],
    window_has: impl Fn(&[u16], usize, u16) -> bool,
) -> usize {
    let (rare, freq) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if rare.is_empty() || freq.is_empty() {
        return 0;
    }
    let (fl, last) = (freq.len(), freq[freq.len() - 1]);
    let (mut k, mut f) = (0, 0);
    for &v in rare {
        if v > last {
            break;
        }
        while f + 8 <= fl && freq[f + 7] < v {
            f += 8;
        }
        let hit = if f + 8 <= fl {
            window_has(freq, f, v)
        } else {
            while f < fl && freq[f] < v {
                f += 1;
            }
            f < fl && freq[f] == v
        };
        if hit {
            out[k] = v;
            k += 1;
        }
    }
    k
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn intersect_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    broadcast_scan(a, b, out, |freq, f, v| unsafe {
        let win = vld1q_u16(freq.as_ptr().add(f));
        vmaxvq_u16(vceqq_u16(win, vdupq_n_u16(v))) != 0
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn intersect_sse2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    broadcast_scan(a, b, out, |freq, f, v| unsafe {
        let win = _mm_loadu_si128(freq.as_ptr().add(f) as *const __m128i);
        _mm_movemask_epi8(_mm_cmpeq_epi16(win, _mm_set1_epi16(v as i16))) != 0
    })
}

/// `a ∪ b` for sorted, unique slices, deduping. Returns the result length.
#[inline]
pub fn array_union(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (v, adv_a, adv_b) = match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => (a[i], true, false),
            std::cmp::Ordering::Greater => (b[j], false, true),
            std::cmp::Ordering::Equal => (a[i], true, true),
        };
        out[k] = v;
        k += 1;
        i += adv_a as usize;
        j += adv_b as usize;
    }
    out[k..k + a.len() - i].copy_from_slice(&a[i..]);
    k += a.len() - i;
    out[k..k + b.len() - j].copy_from_slice(&b[j..]);
    k + b.len() - j
}

/// `a \ b` for sorted, unique slices. `out` may alias `a`. Returns the length.
#[inline]
pub fn array_diff(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (mut i, mut j, mut k) = (0, 0, 0);
    while i < a.len() {
        if j == b.len() || a[i] < b[j] {
            out[k] = a[i];
            k += 1;
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    k
}
