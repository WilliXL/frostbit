//! Bitmap intersection: `dst &= src`, plus a fused-popcount variant.

use crate::api::container::Bitmap;
#[cfg(target_arch = "x86_64")]
use crate::format::BITMAP_WORDS;

/// `dst &= src`, returning the result's population count in one pass.
#[inline]
pub fn and_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            return and_count_avx512(dst, src);
        }
        if is_x86_feature_detected!("avx2") {
            return and_count_avx2(dst, src);
        }
        return and_count_sse2(dst, src);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return crate::ops::simd::common::fold_count_neon(dst, src, |a, b| std::arch::aarch64::vandq_u64(a, b));
    }
    #[allow(unreachable_code)]
    and_count_scalar(dst, src)
}

fn and_count_scalar(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    let mut c = 0;
    for (d, s) in dst.iter_mut().zip(src) {
        *d &= *s;
        c += d.count_ones();
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn and_sse2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m128i>(), src.as_ptr().cast::<__m128i>());
    for i in 0..BITMAP_WORDS / 2 {
        let r = _mm_and_si128(_mm_loadu_si128(dp.add(i)), _mm_loadu_si128(sp.add(i)));
        _mm_storeu_si128(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_avx2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m256i>(), src.as_ptr().cast::<__m256i>());
    for i in 0..BITMAP_WORDS / 4 {
        let r = _mm256_and_si256(_mm256_loadu_si256(dp.add(i)), _mm256_loadu_si256(sp.add(i)));
        _mm256_storeu_si256(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn and_count_sse2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    and_sse2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn and_count_avx2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    and_avx2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_count_avx512(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m512i>(), src.as_ptr().cast::<__m512i>());
    let mut acc = _mm512_setzero_si512();
    for i in 0..BITMAP_WORDS / 8 {
        let r = _mm512_and_si512(_mm512_loadu_si512(dp.add(i)), _mm512_loadu_si512(sp.add(i)));
        _mm512_storeu_si512(dp.add(i), r);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
    }
    _mm512_reduce_add_epi64(acc) as u32
}
