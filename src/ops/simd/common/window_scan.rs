//! Shared SIMD primitive for sorted-`u16` scans: does the `W`-lane window
//! `freq[f..f + W]` contain `v`? Used by both [`super::array_intersect`] (keep
//! hits) and [`super::array_diff`] (keep misses). Callers guarantee
//! `f + W <= freq.len()` and that the relevant CPU feature is present.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn window_has_neon(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::aarch64::*;
    let win = vld1q_u16(freq.as_ptr().add(f));
    vmaxvq_u16(vceqq_u16(win, vdupq_n_u16(v))) != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn window_has_sse2(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::x86_64::*;
    let win = _mm_loadu_si128(freq.as_ptr().add(f).cast());
    _mm_movemask_epi8(_mm_cmpeq_epi16(win, _mm_set1_epi16(v as i16))) != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn window_has_avx2(freq: &[u16], f: usize, v: u16) -> bool {
    use std::arch::x86_64::*;
    let win = _mm256_loadu_si256(freq.as_ptr().add(f).cast());
    _mm256_movemask_epi8(_mm256_cmpeq_epi16(win, _mm256_set1_epi16(v as i16))) != 0
}
