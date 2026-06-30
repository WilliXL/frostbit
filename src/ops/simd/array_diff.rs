//! Sorted `u16` array difference via a SIMD scan.
//!
//! For each value of `a`, gallop a `W`-lane window of `b` forward and test
//! membership at once (the same primitive as [`super::array_intersect`]); keep
//! the value when it is *absent*. Once `a`'s values pass `b`'s last element the
//! rest are copied wholesale. Falls back to a scalar two-pointer. `out` must not
//! alias either input.

/// `a \ b` for sorted, unique slices. Returns the result length.
pub(crate) fn array_diff(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            return diff_avx2(a, b, out);
        }
        return diff_sse2(a, b, out);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return diff_neon(a, b, out);
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn diff_neon(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<8>(a, b, out, |b, f, v| unsafe { super::array_scan::window_has_neon(b, f, v) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn diff_sse2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<8>(a, b, out, |b, f, v| unsafe { super::array_scan::window_has_sse2(b, f, v) })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn diff_avx2(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
    diff_scan::<16>(a, b, out, |b, f, v| unsafe { super::array_scan::window_has_avx2(b, f, v) })
}
