//! Owned `FrozenBitmap`: ingest validation, alignment, equality, view access.

use frostbit::{FrozenBitmap, FrozenBitmapBuilder};

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

#[test]
fn from_bytes_validates() {
    let bm = build(&[1, 2, 3, 70_000]);
    let copy = FrozenBitmap::from_bytes(bm.as_bytes()).expect("valid bytes");
    assert_eq!(copy, bm);

    assert!(FrozenBitmap::from_bytes(&[]).is_none());
    assert!(FrozenBitmap::from_bytes(&[0xFF; 64]).is_none());
    let mut bad = bm.as_bytes().to_vec();
    bad[0] ^= 0xFF;
    assert!(FrozenBitmap::from_bytes(&bad).is_none());
}

#[test]
fn allocations_are_64_aligned() {
    let shapes: Vec<Vec<u32>> = vec![
        vec![],
        vec![42],
        (0..1000).collect(),
        (0..5000).map(|i| i * 2).collect(),
    ];
    for vals in shapes {
        let bm = build(&vals);
        assert_eq!(bm.as_bytes().as_ptr() as usize % 64, 0);
        let copy = FrozenBitmap::from_bytes(bm.as_bytes()).unwrap();
        assert_eq!(copy.as_bytes().as_ptr() as usize % 64, 0);
    }
}

#[test]
fn clone_and_eq() {
    let a = build(&[1, 2, 3]);
    assert_eq!(a, a.clone());
    assert_ne!(a, build(&[1, 2, 4]));
}

#[test]
fn view_access() {
    let bm = build(&[5, 100, 70_000]);
    let v = bm.view();
    assert_eq!(v.len(), 3);
    assert_eq!(v.min(), Some(5));
    assert_eq!(v.max(), Some(70_000));
    // view() also works on ingested copies.
    let copy = FrozenBitmap::from_bytes(bm.as_bytes()).unwrap();
    assert_eq!(copy.view().len(), 3);
}

#[test]
fn owned_read_api() {
    // The owned type queries directly — `len()` is cardinality, not byte length
    // (the old `Deref<[u8]>` footgun where `bm.len()` returned bytes is gone).
    let bm = build(&[1, 2, 3, 70_000]);
    assert_eq!(bm.len(), 4);
    assert!(bm.byte_len() >= 16 && bm.byte_len() != 4);
    assert!(!bm.is_empty());
    assert!(bm.contains(70_000) && !bm.contains(5));
    assert_eq!(bm.min(), Some(1));
    assert_eq!(bm.max(), Some(70_000));
    assert_eq!((&bm).into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 70_000]);

    let empty = FrozenBitmap::empty();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(empty.min().is_none());
}
