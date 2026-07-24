//! SIMD intersection: sorted `u16` arrays and whole-bitmap words.

use super::common::*;
use super::Bitmap;
#[cfg(target_arch = "x86_64")]
use crate::format::BITMAP_WORDS;

// --- sorted u16 arrays ---------------------------------------------------------


/// `a ∩ b` for sorted, unique slices. Returns the result length.
#[inline]
pub fn array_intersect(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    // Disjoint ranges reject in O(1) before any tier — banded/partitioned data
    // annihilates here instead of streaming a full merge (the shuffle-merge has
    // no other early exit). Two compares on lines every tier touches anyway.
    if a[a.len() - 1] < b[0] || b[b.len() - 1] < a[0] {
        return 0;
    }
    let (lo, hi) = if a.len() <= b.len() { (a.len(), b.len()) } else { (b.len(), a.len()) };
    // Heavy skew → galloping binary search (scalar, all targets), but only when
    // its `rare·log2(freq)` work clears the broadcast-scan's `freq/W` linear
    // window advance by a margin (the binary search mispredicts, so it must win
    // comfortably, not just on paper).
    if lo.saturating_mul(hi.ilog2().max(1) as usize) * 8 < hi {
        return intersect_gallop(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Balanced → shuffle-merge; moderate skew → galloping broadcast-scan.
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO {
            return intersect_merge_neon(a, b, out);
        }
        return intersect_neon(a, b, out);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Balanced → shuffle-merge (SSSE3, for the byte-rotate); moderate skew →
        // galloping broadcast-scan.
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO && is_x86_feature_detected!("ssse3") {
            return intersect_merge_sse(a, b, out);
        }
        if is_x86_feature_detected!("avx2") {
            return intersect_avx2(a, b, out);
        }
        return intersect_sse2(a, b, out);
    }
    #[allow(unreachable_code)]
    intersect_scalar(a, b, out)
}

/// Heavily-skewed intersection: for each element of the smaller side, gallop
/// (exponential bracket + binary search) into the larger, advancing a base
/// pointer. O(rare · log gap) — beats scanning the whole larger side.
fn intersect_gallop(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (rare, freq) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let (mut k, mut base) = (0, 0);
    for &v in rare {
        if base >= freq.len() {
            break;
        }
        let mut step = 1;
        while base + step < freq.len() && freq[base + step] < v {
            step <<= 1;
        }
        let lo = base + (step >> 1);
        let hi = (base + step + 1).min(freq.len());
        match freq[lo..hi].binary_search(&v) {
            Ok(p) => {
                out[k] = v;
                k += 1;
                base = lo + p + 1;
            }
            Err(p) => base = lo + p,
        }
    }
    k
}

/// Balanced-array intersection via the CRoaring/Lemire shuffle-merge: compare a
/// full 8-lane block of each side all-pairs (8 rotate-compares → a match mask),
/// emit `a`'s matched lanes, and advance whichever block's max is smaller. One
/// horizontal reduction per 8 elements instead of the broadcast-scan's per
/// element — the win on balanced inputs. A scalar two-pointer finishes the tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn intersect_merge_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    const LANE_BIT: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    let weights = vld1q_u16(LANE_BIT.as_ptr());
    // Keep both 8-lane blocks live across iterations, reloading only the side we
    // advance (the caller guarantees na, nb >= 8).
    let mut va = vld1q_u16(a.as_ptr());
    let mut vb = vld1q_u16(b.as_ptr());
    loop {
        // Per lane of `va`: does it equal any lane of `vb`? All 8 relative
        // offsets via rotations of BOTH operands (Díez-Cañas): `va` and
        // `rot4(va)` each meet 4 rotations of `vb` — offsets {0..3} ∪ {4..7} —
        // then the rot4 group's mask is un-rotated. 5 permutes instead of 7.
        let vb1 = vextq_u16::<1>(vb, vb);
        let vb2 = vextq_u16::<2>(vb, vb);
        let vb3 = vextq_u16::<3>(vb, vb);
        let va4 = vextq_u16::<4>(va, va);
        let lo = vorrq_u16(
            vorrq_u16(vceqq_u16(va, vb), vceqq_u16(va, vb1)),
            vorrq_u16(vceqq_u16(va, vb2), vceqq_u16(va, vb3)),
        );
        let hi = vorrq_u16(
            vorrq_u16(vceqq_u16(va4, vb), vceqq_u16(va4, vb1)),
            vorrq_u16(vceqq_u16(va4, vb2), vceqq_u16(va4, vb3)),
        );
        let m = vorrq_u16(lo, vextq_u16::<4>(hi, hi));
        // Fold the per-lane mask to an 8-bit set, then emit matched `a` lanes,
        // compacted branchlessly by table shuffle (a per-hit loop is fine at
        // sparse overlap but stalls when hits are dense).
        let bits = vaddvq_u16(vandq_u16(m, weights)) as usize;
        // Zero-hit blocks skip the emit entirely — at sparse overlap this
        // branch is almost always false (predicted free); at dense overlap it
        // is almost always true and the branchless table compaction runs.
        if bits == 0 {
        } else if k + 8 <= out.len() {
            let shuf = vld1q_u8(COMPACT[bits].as_ptr());
            let packed = vqtbl1q_u8(vreinterpretq_u8_u16(va), shuf);
            vst1q_u8(out.as_mut_ptr().add(k).cast(), packed);
            k += (bits as u32).count_ones() as usize;
        } else {
            let mut b = bits as u32;
            // Bound the tail against `out.len()`. On sorted-unique inputs the
            // total match count is ≤ out.len(), so this never truncates a valid
            // result; it prevents an out-of-bounds write if the merge is ever
            // handed unsorted/duplicate data (defense-in-depth behind the
            // sortedness validation in `FrozenBitmapView::from_bytes`).
            while b != 0 && k < out.len() {
                let i = b.trailing_zeros() as usize;
                *out.get_unchecked_mut(k) = *a.get_unchecked(ia + i);
                k += 1;
                b &= b - 1;
            }
        }
        // Advance the block whose max is smaller (both on a tie); take maxes from
        // the live registers, and reload only the advanced side.
        let amax = vgetq_lane_u16::<7>(va);
        let bmax = vgetq_lane_u16::<7>(vb);
        if amax <= bmax {
            ia += 8;
            if ia + 8 > na {
                break;
            }
            va = vld1q_u16(a.as_ptr().add(ia));
        }
        if bmax <= amax {
            ib += 8;
            if ib + 8 > nb {
                break;
            }
            vb = vld1q_u16(b.as_ptr().add(ib));
        }
    }
    k + intersect_scalar(&a[ia..], &b[ib..], &mut out[k..])
}

/// Scalar two-pointer reference (and fallback for non-SIMD targets).
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

/// Broadcast-scan skeleton. `window_has(freq, f, v)` reports whether `v` is in
/// the `W`-lane window `freq[f..f + W]`; it is only called when `f + W <= len`.
#[inline(always)]
fn broadcast_scan<const W: usize>(
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
        while f + W <= fl && freq[f + W - 1] < v {
            f += W;
        }
        let hit = if f + W <= fl {
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
    broadcast_scan::<8>(a, b, out, |freq, f, v| unsafe { window_has_neon(freq, f, v) })
}

/// SSSE3 twin of [`intersect_merge_neon`]: the same shuffle-merge, with
/// `_mm_alignr_epi8` for the lane rotate (direction is irrelevant — all 8
/// rotations are OR-ed) and `packs`+`movemask` for the lane mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn intersect_merge_sse(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    let mut va = _mm_loadu_si128(a.as_ptr().cast());
    let mut vb = _mm_loadu_si128(b.as_ptr().cast());
    loop {
        // All-pairs via rotations of BOTH operands (see intersect_merge_neon).
        let vb1 = _mm_alignr_epi8::<2>(vb, vb);
        let vb2 = _mm_alignr_epi8::<4>(vb, vb);
        let vb3 = _mm_alignr_epi8::<6>(vb, vb);
        let va4 = _mm_alignr_epi8::<8>(va, va);
        let lo = _mm_or_si128(
            _mm_or_si128(_mm_cmpeq_epi16(va, vb), _mm_cmpeq_epi16(va, vb1)),
            _mm_or_si128(_mm_cmpeq_epi16(va, vb2), _mm_cmpeq_epi16(va, vb3)),
        );
        let hi = _mm_or_si128(
            _mm_or_si128(_mm_cmpeq_epi16(va4, vb), _mm_cmpeq_epi16(va4, vb1)),
            _mm_or_si128(_mm_cmpeq_epi16(va4, vb2), _mm_cmpeq_epi16(va4, vb3)),
        );
        let m = _mm_or_si128(lo, _mm_alignr_epi8::<8>(hi, hi));
        // Saturating-pack the 8 u16 lanes to bytes (0xFFFF→0xFF, 0→0), then a
        // byte movemask gives the 8-bit lane set; emit via branchless
        // table-shuffle compaction (loop only at the output boundary).
        let bits = (_mm_movemask_epi8(_mm_packs_epi16(m, m)) & 0xFF) as usize;
        // See the NEON twin: zero-hit blocks skip; dense blocks compact
        // branchlessly.
        if bits == 0 {
        } else if k + 8 <= out.len() {
            let shuf = _mm_loadu_si128(COMPACT[bits].as_ptr().cast());
            let packed = _mm_shuffle_epi8(va, shuf);
            _mm_storeu_si128(out.as_mut_ptr().add(k).cast(), packed);
            k += (bits as u32).count_ones() as usize;
        } else {
            let mut b = bits as u32;
            // Bound the tail against `out.len()`. On sorted-unique inputs the
            // total match count is ≤ out.len(), so this never truncates a valid
            // result; it prevents an out-of-bounds write if the merge is ever
            // handed unsorted/duplicate data (defense-in-depth behind the
            // sortedness validation in `FrozenBitmapView::from_bytes`).
            while b != 0 && k < out.len() {
                let i = b.trailing_zeros() as usize;
                *out.get_unchecked_mut(k) = *a.get_unchecked(ia + i);
                k += 1;
                b &= b - 1;
            }
        }
        let amax = _mm_extract_epi16::<7>(va) as u16;
        let bmax = _mm_extract_epi16::<7>(vb) as u16;
        if amax <= bmax {
            ia += 8;
            if ia + 8 > na {
                break;
            }
            va = _mm_loadu_si128(a.as_ptr().add(ia).cast());
        }
        if bmax <= amax {
            ib += 8;
            if ib + 8 > nb {
                break;
            }
            vb = _mm_loadu_si128(b.as_ptr().add(ib).cast());
        }
    }
    k + intersect_scalar(&a[ia..], &b[ib..], &mut out[k..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn intersect_sse2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    broadcast_scan::<8>(a, b, out, |freq, f, v| unsafe { window_has_sse2(freq, f, v) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersect_avx2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    broadcast_scan::<16>(a, b, out, |freq, f, v| unsafe { window_has_avx2(freq, f, v) })
}
// --- whole-bitmap words --------------------------------------------------------
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
        return fold_count_neon(dst, src, |a, b| std::arch::aarch64::vandq_u64(a, b));
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
