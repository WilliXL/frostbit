//! Regression tests for the correctness / memory-safety bugs fixed after the
//! Round-2 audit (see `API_AUDIT.md` §0/§0′). Each test fails on the pre-fix
//! code and passes after the fix.

use frostbit::{difference_fast, union_fast, FrozenBitmap, FrozenBitmapBuilder};

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

/// BUG-1: `union_promotes(n, _)` underflowed `85·n·(n+1) − 86·n − 170` at `n = 1`
/// (every key present in exactly one input), panicking in debug. A single-input
/// union must round-trip the input unchanged.
#[test]
fn bug1_single_input_union_does_not_panic() {
    let a = build(&(0..2000).collect::<Vec<_>>());
    let out = union_fast(&[a.view()]);
    assert_eq!(
        out.view().iter().collect::<Vec<_>>(),
        (0..2000).collect::<Vec<_>>()
    );
}

/// BUG-3: a dense (`card > 4096`) run accumulator minus a bitmap subtrahend hit
/// `st.runs + usize::MAX` in `diff_apply`, overflowing (debug panic / release
/// `unreachable!`). Two dense keys defeat the trivial gate so the diff runs
/// partner-major, which is where `diff_apply` (not the safe `diff_key`) drives.
#[test]
fn bug3_dense_run_minus_bitmap_partner_major() {
    use std::collections::BTreeSet;
    // Two keys, each a dense run in `a` and a bitmap in `b`; > 16 KiB total ⇒
    // non-trivial ⇒ partner-major diff.
    let a_vals: Vec<u32> = (0..10_000).chain(65_536..75_536).collect();
    let b_vals: Vec<u32> = (0u32..16_384)
        .filter(|v| v.is_multiple_of(2))
        .chain((65_536u32..81_920).filter(|v| v.is_multiple_of(2)))
        .collect();
    let a = build(&a_vals);
    let b = build(&b_vals);

    let got: Vec<u32> = difference_fast(&[a.view(), b.view()])
        .view()
        .iter()
        .collect();

    let removed: BTreeSet<u32> = b_vals.iter().copied().collect();
    let want: Vec<u32> = a_vals
        .iter()
        .copied()
        .filter(|v| !removed.contains(v))
        .collect();
    assert_eq!(got, want);
}

// BUG-2 is closed at the parse boundary: the unsorted/duplicate array input that
// overran `intersect_merge`'s output slot is now rejected by `from_bytes` (see the
// `hostile::rejects_unsorted_array` test), so the kernel never sees it. The
// kernel's scalar-tail `k < out.len()` bound remains as defense-in-depth and can
// no longer be reached through the public API.

/// Crafted-byte tests for the `from_bytes` validation gaps (SAFE-1/4/8/13).
/// Gated on `internals` for the wire-format offset helpers. Each corrupts one
/// field of an otherwise-valid buffer and asserts rejection.
#[cfg(feature = "internals")]
mod hostile {
    use frostbit::format::{data_section_off, read_u64, write_u16, write_u64};
    use frostbit::{FrozenBitmap, FrozenBitmapBuilder};

    fn standard(values: &[u32]) -> FrozenBitmap {
        let mut b = FrozenBitmapBuilder::new();
        b.extend_sorted(values.iter().copied());
        b.finish_standard()
    }

    /// SAFE-1: an array payload whose lows are not strictly ascending used to be
    /// accepted, then produced wrong answers / an OOB write in `intersect_merge`.
    #[test]
    fn rejects_unsorted_array() {
        let bm = standard(&(0..100).map(|i| i * 3).collect::<Vec<_>>());
        let mut bytes = bm.as_bytes().to_vec();
        let base = data_section_off(1, false); // single array container, no bitmap
        write_u16(&mut bytes, base, 9999); // first low now exceeds the rest
        assert!(FrozenBitmap::from_bytes(&bytes).is_none());
    }

    /// SAFE-8: a run container claiming `nr = 0` used to be accepted, then
    /// OOB-read in `max()` / `iter()` (via `nr - 1` and `start + 2`).
    #[test]
    fn rejects_zero_run_count() {
        let bm = standard(&(0..1000).collect::<Vec<_>>()); // one run, nr = 1
        let mut bytes = bm.as_bytes().to_vec();
        let base = data_section_off(1, false);
        write_u16(&mut bytes, base, 0); // nr = 0
        assert!(FrozenBitmap::from_bytes(&bytes).is_none());
    }

    /// SAFE-8: a run whose `Σ(len + 1)` disagrees with the index cardinality.
    #[test]
    fn rejects_run_cardinality_mismatch() {
        let bm = standard(&(0..1000).collect::<Vec<_>>()); // run (0, len 999), card 1000
        let mut bytes = bm.as_bytes().to_vec();
        let base = data_section_off(1, false);
        write_u16(&mut bytes, base + 4, 500); // shrink len ⇒ Σ(len+1) = 501 ≠ 1000
        assert!(FrozenBitmap::from_bytes(&bytes).is_none());
    }

    /// SAFE-4: a run with `start + len > 0xFFFF` (would wrap `u16` in `Run::end`).
    #[test]
    fn rejects_run_out_of_range() {
        let bm = standard(&(0..1000).collect::<Vec<_>>());
        let mut bytes = bm.as_bytes().to_vec();
        let base = data_section_off(1, false);
        write_u16(&mut bytes, base + 2, 0xFFF0); // start
        write_u16(&mut bytes, base + 4, 0xFFFF); // len ⇒ end = 0x1FFEF
        assert!(FrozenBitmap::from_bytes(&bytes).is_none());
    }

    /// SAFE-13: a bitmap whose set-bit count disagrees with its declared card
    /// used to be accepted, inflating `size_hint` toward `usize::MAX`.
    #[test]
    fn rejects_bitmap_popcount_mismatch() {
        let bm = standard(&(0..5000).map(|i| i * 2).collect::<Vec<_>>()); // bitmap
        let mut bytes = bm.as_bytes().to_vec();
        let base = data_section_off(1, true); // has_bitmap ⇒ 64-aligned data section
        let w = read_u64(&bytes, base);
        write_u64(&mut bytes, base, w ^ 1); // flip one bit ⇒ popcount ≠ card
        assert!(FrozenBitmap::from_bytes(&bytes).is_none());
    }

    /// SAFE-2 (positive side): the owned constructor copies into an aligned buffer
    /// first, so it accepts a source at any alignment and stays op-safe.
    #[test]
    fn owned_from_bytes_accepts_unaligned_source() {
        let bm = standard(&(0..5000).map(|i| i * 2).collect::<Vec<_>>());
        let mut prefixed = vec![0u8]; // shift the payload off the Vec's base
        prefixed.extend_from_slice(bm.as_bytes());
        let copy = FrozenBitmap::from_bytes(&prefixed[1..]).expect("owned copy realigns");
        assert_eq!(copy.view().len(), bm.view().len());
    }
}
