//! Population count of a bitmap.
//!
//! x86 below AVX-512 uses the scalar `POPCNT` instruction (`count_ones`), which
//! is already optimal in a loop; only AVX-512 VPOPCNTDQ beats it. aarch64 has
//! no scalar popcount, so it uses the NEON `CNT` with a widening reduction.

use super::Bitmap;
use crate::format::BITMAP_WORDS;

#[inline]
pub(crate) fn popcount(b: &Bitmap) -> u32 {
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
