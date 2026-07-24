//! Public set-op API (`*_fast` / `*_compact`) over the stable surface only.
#![cfg(feature = "roaring")]

use frostbit::{
    difference_compact, difference_fast, intersect_compact, intersect_fast, union_compact,
    union_fast, FrozenBitmap,
};
use roaring::RoaringBitmap;

fn fz(values: &[u32]) -> FrozenBitmap {
    FrozenBitmap::from_roaring(&values.iter().copied().collect::<RoaringBitmap>())
}

#[test]
fn fast_ops_match_roaring() {
    let a: Vec<u32> = (0..50_000).map(|i| i * 3).collect();
    let b: Vec<u32> = (0..50_000).map(|i| i * 5).collect();
    let fa = fz(&a);
    let fb = fz(&b);
    let (ra, rb): (RoaringBitmap, RoaringBitmap) =
        (a.iter().copied().collect(), b.iter().copied().collect());

    let got = intersect_fast(&[fa.view(), fb.view()]).to_roaring();
    assert_eq!(got, &ra & &rb);

    let got = union_fast(&[fa.view(), fb.view()]).to_roaring();
    assert_eq!(got, &ra | &rb);

    let got = difference_fast(&[fa.view(), fb.view()]).to_roaring();
    assert_eq!(got, &ra - &rb);
}

#[test]
fn fast_op_results_are_reusable_views() {
    // _fast output is op-ready: feed an op result straight into another op.
    let a = fz(&[1, 2, 3, 4, 5, 6]);
    let b = fz(&[2, 4, 6, 8]);
    let c = fz(&[4, 5, 6, 7]);
    let ab = union_fast(&[a.view(), b.view()]); // {1,2,3,4,5,6,8}
    let got = intersect_fast(&[ab.view(), c.view()]); // ∩ {4,5,6,7} = {4,5,6}
    assert_eq!(got.view().iter().collect::<Vec<_>>(), vec![4, 5, 6]);
}

#[test]
fn compact_ops_hold_the_same_set_and_are_no_larger() {
    let a: Vec<u32> = (0..50_000).map(|i| i * 3).collect();
    let b: Vec<u32> = (0..50_000).map(|i| i * 5).collect();
    let (fa, fb) = (fz(&a), fz(&b));
    let v = [fa.view(), fb.view()];

    for (fast, compact) in [
        (intersect_fast(&v), intersect_compact(&v)),
        (union_fast(&v), union_compact(&v)),
        (difference_fast(&v), difference_compact(&v)),
    ] {
        // Same set (set-equality across encodings)...
        assert_eq!(fast, compact);
        // ...and the compact form is never larger than the op-ready one.
        assert!(compact.byte_len() <= fast.byte_len());
    }
}
