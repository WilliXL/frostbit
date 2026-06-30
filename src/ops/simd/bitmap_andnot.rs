//! Bitmap difference: `dst &= !src`, plus a fused-popcount variant.
//!
//! The hardware "andnot" computes `!a & b`, so to get `dst & !src` the loops
//! pass `(src, dst)`. NEON's `BIC` is `a & !b`, so it takes `(dst, src)`.

use super::Bitmap;
#[cfg(target_arch = "x86_64")]
use crate::format::BITMAP_WORDS;

/// `dst &= !src`.
#[inline]
pub(crate) fn andnot(dst: &mut Bitmap, src: &Bitmap) {
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
pub(crate) fn andnot_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
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
