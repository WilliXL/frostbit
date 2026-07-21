//! Bitmap difference: `dst &= !src`, plus a fused-popcount variant.
//!
//! The hardware "andnot" computes `!a & b`, so to get `dst & !src` the loops
//! pass `(src, dst)`. NEON's `BIC` is `a & !b`, so it takes `(dst, src)`.

use super::Bitmap;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::format::BITMAP_WORDS;

/// `dst = a & !b`, returning the result's population count in one pass — avoids
/// the copy that `load_bitmap(a)` + `andnot(dst, b)` would do (the win on a
/// single dense subtraction).
#[inline]
pub fn andnot_into_count(dst: &mut Bitmap, a: &Bitmap, b: &Bitmap) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return andnot_into_count_avx2(dst, a, b);
        }
        return andnot_into_count_sse2(dst, a, b);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return andnot_into_count_neon(dst, a, b);
    }
    #[allow(unreachable_code)]
    {
        let mut c = 0;
        for ((d, x), y) in dst.iter_mut().zip(a).zip(b) {
            *d = *x & !*y;
            c += d.count_ones();
        }
        c
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn andnot_into_count_neon(dst: &mut Bitmap, a: &Bitmap, b: &Bitmap) -> u32 {
    use std::arch::aarch64::*;
    // Four independent count accumulators: a single `vpadalq` chain is
    // latency-bound at ~3x the streaming rate (measured 305 -> 103 ns).
    let (mut c0, mut c1, mut c2, mut c3) =
        (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
    for i in (0..BITMAP_WORDS).step_by(8) {
        let r0 = vbicq_u64(vld1q_u64(a.as_ptr().add(i)), vld1q_u64(b.as_ptr().add(i)));
        let r1 = vbicq_u64(vld1q_u64(a.as_ptr().add(i + 2)), vld1q_u64(b.as_ptr().add(i + 2)));
        let r2 = vbicq_u64(vld1q_u64(a.as_ptr().add(i + 4)), vld1q_u64(b.as_ptr().add(i + 4)));
        let r3 = vbicq_u64(vld1q_u64(a.as_ptr().add(i + 6)), vld1q_u64(b.as_ptr().add(i + 6)));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn andnot_into_count_sse2(dst: &mut Bitmap, a: &Bitmap, b: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let (dp, ap, bp) =
        (dst.as_mut_ptr().cast::<__m128i>(), a.as_ptr().cast::<__m128i>(), b.as_ptr().cast::<__m128i>());
    for i in 0..BITMAP_WORDS / 2 {
        let r = _mm_andnot_si128(_mm_loadu_si128(bp.add(i)), _mm_loadu_si128(ap.add(i)));
        _mm_storeu_si128(dp.add(i), r);
    }
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn andnot_into_count_avx2(dst: &mut Bitmap, a: &Bitmap, b: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let (dp, ap, bp) =
        (dst.as_mut_ptr().cast::<__m256i>(), a.as_ptr().cast::<__m256i>(), b.as_ptr().cast::<__m256i>());
    for i in 0..BITMAP_WORDS / 4 {
        let r = _mm256_andnot_si256(_mm256_loadu_si256(bp.add(i)), _mm256_loadu_si256(ap.add(i)));
        _mm256_storeu_si256(dp.add(i), r);
    }
    dst.iter().map(|w| w.count_ones()).sum()
}

/// `dst &= !src`.
#[inline]
pub fn andnot(dst: &mut Bitmap, src: &Bitmap) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") {
            return andnot_avx512(dst, src);
        }
        if is_x86_feature_detected!("avx2") {
            return andnot_avx2(dst, src);
        }
        return andnot_sse2(dst, src);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return super::fold_neon(dst, src, |a, b| std::arch::aarch64::vbicq_u64(a, b));
    }
    #[allow(unreachable_code)]
    andnot_scalar(dst, src);
}

/// `dst &= !src`, returning the result's population count in one pass.
#[inline]
pub fn andnot_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            return andnot_count_avx512(dst, src);
        }
        if is_x86_feature_detected!("avx2") {
            return andnot_count_avx2(dst, src);
        }
        return andnot_count_sse2(dst, src);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return super::fold_count_neon(dst, src, |a, b| std::arch::aarch64::vbicq_u64(a, b));
    }
    #[allow(unreachable_code)]
    andnot_count_scalar(dst, src)
}

fn andnot_scalar(dst: &mut Bitmap, src: &Bitmap) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= !*s;
    }
}

fn andnot_count_scalar(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    let mut c = 0;
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= !*s;
        c += d.count_ones();
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn andnot_sse2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m128i>(), src.as_ptr().cast::<__m128i>());
    for i in 0..BITMAP_WORDS / 2 {
        let r = _mm_andnot_si128(_mm_loadu_si128(sp.add(i)), _mm_loadu_si128(dp.add(i)));
        _mm_storeu_si128(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn andnot_avx2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m256i>(), src.as_ptr().cast::<__m256i>());
    for i in 0..BITMAP_WORDS / 4 {
        let r = _mm256_andnot_si256(_mm256_loadu_si256(sp.add(i)), _mm256_loadu_si256(dp.add(i)));
        _mm256_storeu_si256(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn andnot_avx512(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m512i>(), src.as_ptr().cast::<__m512i>());
    for i in 0..BITMAP_WORDS / 8 {
        let r = _mm512_andnot_si512(_mm512_loadu_si512(sp.add(i)), _mm512_loadu_si512(dp.add(i)));
        _mm512_storeu_si512(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn andnot_count_sse2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    andnot_sse2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn andnot_count_avx2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    andnot_avx2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn andnot_count_avx512(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m512i>(), src.as_ptr().cast::<__m512i>());
    let mut acc = _mm512_setzero_si512();
    for i in 0..BITMAP_WORDS / 8 {
        let r = _mm512_andnot_si512(_mm512_loadu_si512(sp.add(i)), _mm512_loadu_si512(dp.add(i)));
        _mm512_storeu_si512(dp.add(i), r);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
    }
    _mm512_reduce_add_epi64(acc) as u32
}
