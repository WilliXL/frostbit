//! SIMD union: sorted `u16` arrays and whole-bitmap words.

use super::common::*;
use super::Bitmap;
#[cfg(target_arch = "x86_64")]
use crate::format::BITMAP_WORDS;

// --- sorted u16 arrays ---------------------------------------------------------
/// `a ∪ b` for sorted, unique slices, deduping. Returns the result length.
pub fn array_union(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (lo, hi) = if a.len() <= b.len() { (a.len(), b.len()) } else { (b.len(), a.len()) };
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO {
            return union_merge_neon(a, b, out);
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO && is_x86_feature_detected!("sse4.1") {
            return union_merge_sse(a, b, out);
        }
    }
    union_scalar(a, b, out)
}

/// Scalar two-pointer merge with dedup (fallback + small inputs).
fn union_scalar(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
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

/// Finish a merge: 3-way scalar merge of the `carry` (high block) with the
/// leftover `a`/`b`, deduping against each other and the last value emitted.
fn union_tail(carry: &[u16], a: &[u16], b: &[u16], out: &mut [u16], mut k: usize) -> usize {
    let (mut p, mut q, mut r) = (0usize, 0usize, 0usize);
    loop {
        let cv = carry.get(p).copied().unwrap_or(u16::MAX);
        let av = a.get(q).copied().unwrap_or(u16::MAX);
        let bv = b.get(r).copied().unwrap_or(u16::MAX);
        let best = cv.min(av).min(bv);
        if best == u16::MAX && p >= carry.len() && q >= a.len() && r >= b.len() {
            return k;
        }
        if k == 0 || out[k - 1] != best {
            out[k] = best;
            k += 1;
        }
        p += (cv == best) as usize;
        q += (av == best) as usize;
        r += (bv == best) as usize;
    }
}

// --- aarch64 ----------------------------------------------------------------

/// Merge two sorted 8-lane vectors into `(low 8, high 8)`, both sorted, via the
/// Inoue–Taura rotate network.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn simd_merge_neon(
    a: std::arch::aarch64::uint16x8_t,
    b: std::arch::aarch64::uint16x8_t,
) -> (std::arch::aarch64::uint16x8_t, std::arch::aarch64::uint16x8_t) {
    use std::arch::aarch64::*;
    let mut tmp = vminq_u16(a, b);
    let mut max = vmaxq_u16(a, b);
    tmp = vextq_u16::<1>(tmp, tmp);
    let mut min = vminq_u16(tmp, max);
    for _ in 0..6 {
        max = vmaxq_u16(tmp, max);
        tmp = vextq_u16::<1>(min, min);
        min = vminq_u16(tmp, max);
    }
    max = vmaxq_u16(tmp, max);
    min = vextq_u16::<1>(min, min);
    (min, max)
}

/// Emit `new`'s lanes that differ from their predecessor (`prev`'s last lane
/// bridges the block boundary), compacted by table shuffle.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn emit_unique_neon(
    prev: std::arch::aarch64::uint16x8_t,
    new: std::arch::aarch64::uint16x8_t,
    weights: std::arch::aarch64::uint16x8_t,
    out: &mut [u16],
    k: usize,
) -> usize {
    use std::arch::aarch64::*;
    let tmp = vextq_u16::<7>(prev, new);
    let eq = vaddvq_u16(vandq_u16(vceqq_u16(tmp, new), weights));
    let uniq = (!eq & 0xFF) as usize;
    if k + 8 <= out.len() {
        let shuf = vld1q_u8(COMPACT[uniq].as_ptr());
        let packed = vqtbl1q_u8(vreinterpretq_u8_u16(new), shuf);
        vst1q_u8(out.as_mut_ptr().add(k).cast(), packed);
        k + uniq.count_ones() as usize
    } else {
        let mut buf = [0u16; 8];
        vst1q_u16(buf.as_mut_ptr(), new);
        let (mut kk, mut u) = (k, uniq as u32);
        while u != 0 {
            let i = u.trailing_zeros() as usize;
            out[kk] = buf[i];
            kk += 1;
            u &= u - 1;
        }
        kk
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn union_merge_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    const LANE_BIT: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    let weights = vld1q_u16(LANE_BIT.as_ptr());
    let (len_a, len_b) = (a.len() / 8, b.len() / 8);
    let (mut vmin, mut vmax) =
        simd_merge_neon(vld1q_u16(a.as_ptr()), vld1q_u16(b.as_ptr()));
    let mut k = emit_unique_neon(vdupq_n_u16(u16::MAX), vmin, weights, out, 0);
    let mut vprev = vmin;
    let (mut i, mut j) = (1usize, 1usize);
    if i < len_a && j < len_b {
        let mut cur_a = *a.get_unchecked(8 * i);
        let mut cur_b = *b.get_unchecked(8 * j);
        let mut v;
        loop {
            if cur_a <= cur_b {
                v = vld1q_u16(a.as_ptr().add(8 * i));
                i += 1;
                if i < len_a {
                    cur_a = *a.get_unchecked(8 * i);
                } else {
                    break;
                }
            } else {
                v = vld1q_u16(b.as_ptr().add(8 * j));
                j += 1;
                if j < len_b {
                    cur_b = *b.get_unchecked(8 * j);
                } else {
                    break;
                }
            }
            (vmin, vmax) = simd_merge_neon(v, vmax);
            k = emit_unique_neon(vprev, vmin, weights, out, k);
            vprev = vmin;
        }
        (vmin, vmax) = simd_merge_neon(v, vmax);
        k = emit_unique_neon(vprev, vmin, weights, out, k);
    }
    let mut buf = [0u16; 8];
    vst1q_u16(buf.as_mut_ptr(), vmax);
    union_tail(&buf, &a[8 * i..], &b[8 * j..], out, k)
}

// --- x86_64 -----------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn simd_merge_sse(
    a: std::arch::x86_64::__m128i,
    b: std::arch::x86_64::__m128i,
) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    let rot1 = |v| _mm_alignr_epi8::<2>(v, v);
    let mut tmp = _mm_min_epu16(a, b);
    let mut max = _mm_max_epu16(a, b);
    tmp = rot1(tmp);
    let mut min = _mm_min_epu16(tmp, max);
    for _ in 0..6 {
        max = _mm_max_epu16(tmp, max);
        tmp = rot1(min);
        min = _mm_min_epu16(tmp, max);
    }
    max = _mm_max_epu16(tmp, max);
    min = rot1(min);
    (min, max)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn emit_unique_sse(
    prev: std::arch::x86_64::__m128i,
    new: std::arch::x86_64::__m128i,
    out: &mut [u16],
    k: usize,
) -> usize {
    use std::arch::x86_64::*;
    let tmp = _mm_alignr_epi8::<14>(new, prev);
    let eq = _mm_movemask_epi8(_mm_packs_epi16(_mm_cmpeq_epi16(tmp, new), _mm_setzero_si128()));
    let uniq = (!eq & 0xFF) as usize;
    if k + 8 <= out.len() {
        let shuf = _mm_loadu_si128(COMPACT[uniq].as_ptr().cast());
        let packed = _mm_shuffle_epi8(new, shuf);
        _mm_storeu_si128(out.as_mut_ptr().add(k).cast(), packed);
        k + uniq.count_ones() as usize
    } else {
        let mut buf = [0u16; 8];
        _mm_storeu_si128(buf.as_mut_ptr().cast(), new);
        let (mut kk, mut u) = (k, uniq as u32);
        while u != 0 {
            let i = u.trailing_zeros() as usize;
            out[kk] = buf[i];
            kk += 1;
            u &= u - 1;
        }
        kk
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn union_merge_sse(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let (len_a, len_b) = (a.len() / 8, b.len() / 8);
    let (mut vmin, mut vmax) =
        simd_merge_sse(_mm_loadu_si128(a.as_ptr().cast()), _mm_loadu_si128(b.as_ptr().cast()));
    let mut k = emit_unique_sse(_mm_set1_epi16(-1), vmin, out, 0);
    let mut vprev = vmin;
    let (mut i, mut j) = (1usize, 1usize);
    if i < len_a && j < len_b {
        let mut cur_a = *a.get_unchecked(8 * i);
        let mut cur_b = *b.get_unchecked(8 * j);
        let mut v;
        loop {
            if cur_a <= cur_b {
                v = _mm_loadu_si128(a.as_ptr().add(8 * i).cast());
                i += 1;
                if i < len_a {
                    cur_a = *a.get_unchecked(8 * i);
                } else {
                    break;
                }
            } else {
                v = _mm_loadu_si128(b.as_ptr().add(8 * j).cast());
                j += 1;
                if j < len_b {
                    cur_b = *b.get_unchecked(8 * j);
                } else {
                    break;
                }
            }
            (vmin, vmax) = simd_merge_sse(v, vmax);
            k = emit_unique_sse(vprev, vmin, out, k);
            vprev = vmin;
        }
        (vmin, vmax) = simd_merge_sse(v, vmax);
        k = emit_unique_sse(vprev, vmin, out, k);
    }
    let mut buf = [0u16; 8];
    _mm_storeu_si128(buf.as_mut_ptr().cast(), vmax);
    union_tail(&buf, &a[8 * i..], &b[8 * j..], out, k)
}

// --- whole-bitmap words --------------------------------------------------------
/// `dst |= src`.
#[inline]
pub fn or(dst: &mut Bitmap, src: &Bitmap) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") {
            return or_avx512(dst, src);
        }
        if is_x86_feature_detected!("avx2") {
            return or_avx2(dst, src);
        }
        return or_sse2(dst, src);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return fold_neon(dst, src, |a, b| std::arch::aarch64::vorrq_u64(a, b));
    }
    #[allow(unreachable_code)]
    or_scalar(dst, src);
}

/// `dst |= src`, returning the result's population count in one pass.
#[inline]
pub fn or_count(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq") {
            return or_count_avx512(dst, src);
        }
        if is_x86_feature_detected!("avx2") {
            return or_count_avx2(dst, src);
        }
        return or_count_sse2(dst, src);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return fold_count_neon(dst, src, |a, b| std::arch::aarch64::vorrq_u64(a, b));
    }
    #[allow(unreachable_code)]
    or_count_scalar(dst, src)
}

fn or_scalar(dst: &mut Bitmap, src: &Bitmap) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

fn or_count_scalar(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    let mut c = 0;
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
        c += d.count_ones();
    }
    c
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn or_sse2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m128i>(), src.as_ptr().cast::<__m128i>());
    for i in 0..BITMAP_WORDS / 2 {
        let r = _mm_or_si128(_mm_loadu_si128(dp.add(i)), _mm_loadu_si128(sp.add(i)));
        _mm_storeu_si128(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn or_avx2(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m256i>(), src.as_ptr().cast::<__m256i>());
    for i in 0..BITMAP_WORDS / 4 {
        let r = _mm256_or_si256(_mm256_loadu_si256(dp.add(i)), _mm256_loadu_si256(sp.add(i)));
        _mm256_storeu_si256(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn or_avx512(dst: &mut Bitmap, src: &Bitmap) {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m512i>(), src.as_ptr().cast::<__m512i>());
    for i in 0..BITMAP_WORDS / 8 {
        let r = _mm512_or_si512(_mm512_loadu_si512(dp.add(i)), _mm512_loadu_si512(sp.add(i)));
        _mm512_storeu_si512(dp.add(i), r);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn or_count_sse2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    or_sse2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn or_count_avx2(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    or_avx2(dst, src);
    dst.iter().map(|w| w.count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn or_count_avx512(dst: &mut Bitmap, src: &Bitmap) -> u32 {
    use std::arch::x86_64::*;
    let (dp, sp) = (dst.as_mut_ptr().cast::<__m512i>(), src.as_ptr().cast::<__m512i>());
    let mut acc = _mm512_setzero_si512();
    for i in 0..BITMAP_WORDS / 8 {
        let r = _mm512_or_si512(_mm512_loadu_si512(dp.add(i)), _mm512_loadu_si512(sp.add(i)));
        _mm512_storeu_si512(dp.add(i), r);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(r));
    }
    _mm512_reduce_add_epi64(acc) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn gen(s: &mut u64, n: usize, spread: u16) -> Vec<u16> {
        let mut set = BTreeSet::new();
        while set.len() < n {
            set.insert((splitmix(s) % spread as u64) as u16);
        }
        set.into_iter().collect()
    }

    fn naive(a: &[u16], b: &[u16]) -> Vec<u16> {
        let mut s: BTreeSet<u16> = a.iter().copied().collect();
        s.extend(b.iter().copied());
        s.into_iter().collect()
    }

    /// `array_union` (every tier) must equal a set union — random sizes and
    /// overlap so the merge network, boundary dedup, and the 3-way scalar tail
    /// are all exercised.
    #[test]
    fn union_matches_naive() {
        let mut s = 0x0F55_1234_u64;
        for _ in 0..4000 {
            let spread = 40 + (splitmix(&mut s) % 6000) as u16;
            let cap = (spread as usize - 1).min(500);
            let n = 1 + splitmix(&mut s) as usize % cap;
            let m = 1 + splitmix(&mut s) as usize % cap;
            let a = gen(&mut s, n, spread);
            let b = gen(&mut s, m, spread);
            let mut out = vec![0u16; a.len() + b.len()];
            let k = array_union(&a, &b, &mut out);
            assert_eq!(&out[..k], naive(&a, &b).as_slice(), "\na={a:?}\nb={b:?}");
        }
    }
}
