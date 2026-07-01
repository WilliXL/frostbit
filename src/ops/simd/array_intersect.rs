//! Sorted `u16` array intersection via a SIMD broadcast scan.
//!
//! For each value of the smaller ("rare") side, gallop a `W`-lane window of the
//! larger ("freq") side forward and test all lanes at once. Falls back to a
//! scalar two-pointer merge. `out` must not alias either input.

/// At or below this size ratio the arrays are balanced enough that the
/// shuffle-merge's 8-at-a-time compare wins (one horizontal reduction per block,
/// not per element). Above it, one side is "rare" — a scan of the larger wins.
const MERGE_MAX_RATIO: usize = 4;

/// `a ∩ b` for sorted, unique slices. Returns the result length.
#[inline]
pub(crate) fn array_intersect(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
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
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    const LANE_BIT: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    let weights = vld1q_u16(LANE_BIT.as_ptr());
    while ia + 8 <= a.len() && ib + 8 <= b.len() {
        let va = vld1q_u16(a.as_ptr().add(ia));
        let vb = vld1q_u16(b.as_ptr().add(ib));
        // Per lane of `va`: does it equal any lane of `vb`? OR of 8 rotations.
        let mut m = vceqq_u16(va, vb);
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<1>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<2>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<3>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<4>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<5>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<6>(vb, vb)));
        m = vorrq_u16(m, vceqq_u16(va, vextq_u16::<7>(vb, vb)));
        // Fold the per-lane mask to an 8-bit set, then emit matched `a` lanes.
        let mut bits = vaddvq_u16(vandq_u16(m, weights));
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            *out.get_unchecked_mut(k) = *a.get_unchecked(ia + i);
            k += 1;
            bits &= bits - 1;
        }
        let amax = *a.get_unchecked(ia + 7);
        let bmax = *b.get_unchecked(ib + 7);
        ia += usize::from(amax <= bmax) << 3;
        ib += usize::from(bmax <= amax) << 3;
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
    broadcast_scan::<8>(a, b, out, |freq, f, v| unsafe { super::array_scan::window_has_neon(freq, f, v) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn intersect_sse2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    broadcast_scan::<8>(a, b, out, |freq, f, v| unsafe { super::array_scan::window_has_sse2(freq, f, v) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersect_avx2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    broadcast_scan::<16>(a, b, out, |freq, f, v| unsafe { super::array_scan::window_has_avx2(freq, f, v) })
}
