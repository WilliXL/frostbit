//! Sorted `u16` array difference via a SIMD scan.
//!
//! For each value of `a`, gallop a `W`-lane window of `b` forward and test
//! membership at once (the same primitive as [`super::array_intersect`]); keep
//! the value when it is *absent*. Once `a`'s values pass `b`'s last element the
//! rest are copied wholesale. Falls back to a scalar two-pointer. `out` must not
//! alias either input.

use crate::ops::simd::common::{COMPACT, MERGE_MAX_RATIO};


/// `a \ b` for sorted, unique slices. Returns the result length.
pub fn array_diff(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    let (lo, hi) = if a.len() <= b.len() { (a.len(), b.len()) } else { (b.len(), a.len()) };
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    unsafe {
        // Balanced → shuffle-merge; skewed → the galloping broadcast-scan.
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO {
            return diff_merge_neon(a, b, out);
        }
        return diff_neon(a, b, out);
    }
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        if lo >= 8 && hi <= lo * MERGE_MAX_RATIO && is_x86_feature_detected!("ssse3") {
            return diff_merge_sse(a, b, out);
        }
        if is_x86_feature_detected!("avx2") {
            return diff_avx2(a, b, out);
        }
        return diff_sse2(a, b, out);
    }
    #[allow(unreachable_code)]
    diff_scalar(a, b, out)
}

/// Scalar two-pointer reference (and fallback for non-SIMD targets).
fn diff_scalar(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
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

/// Scan skeleton: emit each `a` value not found in `b`. `window_has(b, f, v)`
/// reports whether `v` is in the `W`-lane window `b[f..f + W]`; it is only
/// called when `f + W <= b.len()`.
#[inline(always)]
fn diff_scan<const W: usize>(
    a: &[u16],
    b: &[u16],
    out: &mut [u16],
    window_has: impl Fn(&[u16], usize, u16) -> bool,
) -> usize {
    let al = a.len();
    if b.is_empty() {
        out[..al].copy_from_slice(a);
        return al;
    }
    let (bl, last) = (b.len(), b[b.len() - 1]);
    let (mut k, mut i, mut f) = (0, 0, 0);
    while i < al {
        let v = a[i];
        if v > last {
            // Every remaining value of `a` is past `b`'s end — all kept.
            let rem = al - i;
            out[k..k + rem].copy_from_slice(&a[i..]);
            return k + rem;
        }
        while f + W <= bl && b[f + W - 1] < v {
            f += W;
        }
        let hit = if f + W <= bl {
            window_has(b, f, v)
        } else {
            while f < bl && b[f] < v {
                f += 1;
            }
            f < bl && b[f] == v
        };
        if !hit {
            out[k] = v;
            k += 1;
        }
        i += 1;
    }
    k
}

#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon")]
unsafe fn diff_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<8>(a, b, out, |b, f, v| unsafe { crate::ops::simd::common::window_has_neon(b, f, v) })
}

/// Finish a difference merge once the SIMD loop breaks on an exhausted side.
/// `a_emitted` = the current `a` block was already emitted (a advanced). If not,
/// it is still pending with accumulated `matched`, and the leftover `b[ib..]`
/// (< 8) has not been tested against it yet.
fn diff_tail(
    a: &[u16],
    ia: usize,
    b: &[u16],
    ib: usize,
    matched: u32,
    a_emitted: bool,
    out: &mut [u16],
) -> usize {
    if a_emitted {
        return diff_scalar(&a[ia..], &b[ib..], out);
    }
    let bt = &b[ib..];
    let mut k = 0;
    for lane in 0..8 {
        if matched & (1 << lane) == 0 {
            let x = a[ia + lane];
            if bt.binary_search(&x).is_err() {
                out[k] = x;
                k += 1;
            }
        }
    }
    k + diff_scalar(&a[ia + 8..], bt, &mut out[k..])
}

/// Balanced-array difference via the shuffle-merge. Unlike intersection, the
/// per-`a`-lane match mask is *accumulated* across every `b` block the `a` block
/// spans (a value absent from one `b` block may match a later one), and `a`'s
/// *un*-matched lanes are emitted only when `a` advances — by which point it has
/// seen every `b` value ≤ its max. A scalar two-pointer finishes the tail.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[target_feature(enable = "neon")]
unsafe fn diff_merge_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::aarch64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    const LANE_BIT: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    let weights = vld1q_u16(LANE_BIT.as_ptr());
    let mut va = vld1q_u16(a.as_ptr());
    let mut vb = vld1q_u16(b.as_ptr());
    let mut matched = 0u32;
    loop {
        // All-pairs via rotations of BOTH operands (see intersect_merge_neon).
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
        matched |= vaddvq_u16(vandq_u16(m, weights)) as u32;
        let amax = vgetq_lane_u16::<7>(va);
        let bmax = vgetq_lane_u16::<7>(vb);
        if amax <= bmax {
            // Emit `va`'s un-matched lanes, compacted 8-at-a-time by table shuffle
            // (most lanes survive a difference, so a per-lane loop is the cost).
            let un = (!matched & 0xFF) as usize;
            if k + 8 <= out.len() {
                let shuf = vld1q_u8(COMPACT[un].as_ptr());
                let packed = vqtbl1q_u8(vreinterpretq_u8_u16(va), shuf);
                vst1q_u8(out.as_mut_ptr().add(k).cast(), packed);
                k += (un as u32).count_ones() as usize;
            } else {
                let mut u = un as u32;
                while u != 0 {
                    let i = u.trailing_zeros() as usize;
                    *out.get_unchecked_mut(k) = *a.get_unchecked(ia + i);
                    k += 1;
                    u &= u - 1;
                }
            }
            matched = 0;
            ia += 8;
            if ia + 8 > na {
                if amax == bmax {
                    ib += 8;
                }
                return k + diff_scalar(&a[ia..], &b[ib..], &mut out[k..]);
            }
            va = vld1q_u16(a.as_ptr().add(ia));
        }
        if bmax <= amax {
            ib += 8;
            if ib + 8 > nb {
                return k + diff_tail(a, ia, b, ib, matched, amax <= bmax, &mut out[k..]);
            }
            vb = vld1q_u16(b.as_ptr().add(ib));
        }
    }
}

/// SSSE3 twin of [`diff_merge_neon`].
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "ssse3")]
unsafe fn diff_merge_sse(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    use std::arch::x86_64::*;
    let (na, nb) = (a.len(), b.len());
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    let mut va = _mm_loadu_si128(a.as_ptr().cast());
    let mut vb = _mm_loadu_si128(b.as_ptr().cast());
    let mut matched = 0u32;
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
        matched |= (_mm_movemask_epi8(_mm_packs_epi16(m, m)) & 0xFF) as u32;
        let amax = _mm_extract_epi16::<7>(va) as u16;
        let bmax = _mm_extract_epi16::<7>(vb) as u16;
        if amax <= bmax {
            let un = (!matched & 0xFF) as usize;
            if k + 8 <= out.len() {
                let shuf = _mm_loadu_si128(COMPACT[un].as_ptr().cast());
                let packed = _mm_shuffle_epi8(va, shuf);
                _mm_storeu_si128(out.as_mut_ptr().add(k).cast(), packed);
                k += (un as u32).count_ones() as usize;
            } else {
                let mut u = un as u32;
                while u != 0 {
                    let i = u.trailing_zeros() as usize;
                    *out.get_unchecked_mut(k) = *a.get_unchecked(ia + i);
                    k += 1;
                    u &= u - 1;
                }
            }
            matched = 0;
            ia += 8;
            if ia + 8 > na {
                if amax == bmax {
                    ib += 8;
                }
                return k + diff_scalar(&a[ia..], &b[ib..], &mut out[k..]);
            }
            va = _mm_loadu_si128(a.as_ptr().add(ia).cast());
        }
        if bmax <= amax {
            ib += 8;
            if ib + 8 > nb {
                return k + diff_tail(a, ia, b, ib, matched, amax <= bmax, &mut out[k..]);
            }
            vb = _mm_loadu_si128(b.as_ptr().add(ib).cast());
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "sse2")]
unsafe fn diff_sse2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<8>(a, b, out, |b, f, v| unsafe { crate::ops::simd::common::window_has_sse2(b, f, v) })
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
#[target_feature(enable = "avx2")]
unsafe fn diff_avx2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<16>(a, b, out, |b, f, v| unsafe { crate::ops::simd::common::window_has_avx2(b, f, v) })
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
        a.iter().copied().filter(|x| b.binary_search(x).is_err()).collect()
    }

    /// `array_diff` (every dispatch tier) must equal a naive filter — random
    /// sizes and overlap so the merge's accumulation + both tail branches (`a`
    /// exhausts vs `b` exhausts with a block still pending) are all exercised.
    #[test]
    fn diff_matches_naive() {
        let mut s = 0xD1FF_0FF5_u64;
        for _ in 0..4000 {
            let spread = 40 + (splitmix(&mut s) % 6000) as u16;
            let cap = (spread as usize - 1).min(500);
            let n = 1 + splitmix(&mut s) as usize % cap;
            let m = 1 + splitmix(&mut s) as usize % cap;
            let a = gen(&mut s, n, spread);
            let b = gen(&mut s, m, spread);
            let mut out = vec![0u16; a.len() + 1];
            let k = array_diff(&a, &b, &mut out);
            assert_eq!(&out[..k], naive(&a, &b).as_slice(), "\na={a:?}\nb={b:?}");
        }
    }
}
