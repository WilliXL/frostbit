//! Sorted `u16` array difference (scalar two-pointer; see [`super::array_union`]
//! for why these merges stay scalar).

/// `a \ b` for sorted, unique slices. `out` may alias `a`. Returns the length.
pub(crate) fn array_diff(a: &[u16], b: &[u16], out: &mut [u16]) -> usize {
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
