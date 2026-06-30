//! Sorted `u16` array union (deduping merge).
//!
//! A two-pointer merge has data-dependent control flow that does not vectorize
//! cleanly, so this is scalar on every target; dense unions take the bitmap
//! path in the kernels instead.

/// `a ∪ b` for sorted, unique slices, deduping. Returns the result length.
pub(crate) fn array_union(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
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
