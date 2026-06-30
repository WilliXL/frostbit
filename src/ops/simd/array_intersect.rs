//! Sorted `u16` array intersection via a SIMD broadcast scan.
//!
//! For each value of the smaller ("rare") side, gallop a `W`-lane window of the
//! larger ("freq") side forward and test all lanes at once. Falls back to a
//! scalar two-pointer merge. `out` must not alias either input.

/// `a ∩ b` for sorted, unique slices. Returns the result length.
#[inline]
pub(crate) fn array_intersect(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return intersect_avx2(a, b, out);
        }
        return intersect_sse2(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return intersect_neon(a, b, out);
    }
    #[allow(unreachable_code)]
    intersect_scalar(a, b, out)
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
