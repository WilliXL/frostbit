//! NEON word-fold skeletons shared by the whole-bitmap kernels.

use crate::container::Bitmap;
use crate::format::BITMAP_WORDS;

// --- shared aarch64 NEON word loops -----------------------------------------
//
// NEON is baseline on aarch64, so its intrinsics can be called from these
// generic helpers (the `combine` closure is monomorphized inline). x86 cannot
// do this for non-baseline features, so its loops are written out per op.

/// `dst = combine(dst, src)` word-by-word (two `u64` lanes per step).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn fold_neon(
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
pub(crate) unsafe fn fold_count_neon(
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
