//! `FrozenBitmapView` parse + len/min/max over both encodings, public API only.

use frostbit::{FrozenBitmapBuilder, FrozenBitmapView};

fn build(values: &[u32]) -> frostbit::FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

fn check(values: &[u32]) {
    let bm = build(values);
    let v = FrozenBitmapView::from_bytes(bm.as_bytes()).expect("valid");
    assert_eq!(v.len(), values.len() as u64, "len for {} values", values.len());
    assert_eq!(v.is_empty(), values.is_empty());
    assert_eq!(v.min(), values.first().copied());
    assert_eq!(v.max(), values.last().copied());
}

#[test]
fn empty() {
    check(&[]);
    let bm = build(&[]);
    let v = FrozenBitmapView::from_bytes(bm.as_bytes()).unwrap();
    assert_eq!(v.num_containers(), 0);
    assert_eq!(v.min(), None);
    assert_eq!(v.max(), None);
}

#[test]
fn across_encodings_and_container_types() {
    check(&[42]); // inline
    check(&[0, 65_536, 131_072]); // inline, multi-key
    check(&(0..100).map(|i| i * 3).collect::<Vec<_>>()); // standard: array
    check(&(0..1000).collect::<Vec<_>>()); // standard: run
    check(&(0..5000).map(|i| i * 2).collect::<Vec<_>>()); // standard: bitmap
}

#[test]
fn inline_vs_standard_detection() {
    let inline = build(&[5, 70_000]);
    let v = FrozenBitmapView::from_bytes(inline.as_bytes()).unwrap();
    assert!(v.is_inline());
    assert_eq!(v.num_containers(), 0);

    let standard = build(&(0..1000).collect::<Vec<_>>());
    let v = FrozenBitmapView::from_bytes(standard.as_bytes()).unwrap();
    assert!(!v.is_inline());
    assert_eq!(v.num_containers(), 1);
}

#[test]
fn multi_container_min_max() {
    let mut vals = Vec::new();
    for base in [0u32, 65_536, 1_000_000] {
        for i in 0..50 {
            vals.push(base + i * 10);
        }
    }
    let bm = build(&vals);
    let v = FrozenBitmapView::from_bytes(bm.as_bytes()).unwrap();
    assert!(!v.is_inline()); // 150 clustered values: standard wins
    assert_eq!(v.num_containers(), 3);
    assert_eq!(v.min(), Some(0));
    assert_eq!(v.max(), Some(1_000_000 + 49 * 10));
    assert_eq!(v.len(), 150);
}

#[test]
fn boundary_values() {
    check(&[0, u32::MAX / 2, u32::MAX]);
    check(&[0xFFFE, 0xFFFF, 0x1_0000, 0x1_0001]); // container boundary
}

#[test]
fn rejects_garbage() {
    assert!(FrozenBitmapView::from_bytes(&[]).is_none());
    assert!(FrozenBitmapView::from_bytes(&[0u8; 8]).is_none());
    assert!(FrozenBitmapView::from_bytes(&[0xFF; 64]).is_none());

    // Corrupt the magic of a valid (inline) bitmap.
    let bm = build(&[1, 2, 3]);
    let mut bad = bm.as_bytes().to_vec();
    bad[0] ^= 0xFF;
    assert!(FrozenBitmapView::from_bytes(&bad).is_none());

    // Truncated standard payload.
    let bm = build(&(0..5000).map(|i| i * 2).collect::<Vec<_>>());
    let truncated = &bm.as_bytes()[..bm.byte_len() - 100];
    assert!(FrozenBitmapView::from_bytes(truncated).is_none());
}

#[test]
fn rejects_bad_inline() {
    // Descending values: "FI", count=2, [100, 50].
    let mut buf = vec![b'F', b'I', 2, 0];
    buf.extend_from_slice(&100u32.to_le_bytes());
    buf.extend_from_slice(&50u32.to_le_bytes());
    assert!(FrozenBitmapView::from_bytes(&buf).is_none());

    // Duplicate values.
    let mut buf = vec![b'F', b'I', 2, 0];
    buf.extend_from_slice(&100u32.to_le_bytes());
    buf.extend_from_slice(&100u32.to_le_bytes());
    assert!(FrozenBitmapView::from_bytes(&buf).is_none());

    // Count overruns the buffer.
    let mut buf = vec![b'F', b'I', 3, 0];
    buf.extend_from_slice(&1u32.to_le_bytes());
    assert!(FrozenBitmapView::from_bytes(&buf).is_none());

    // Count 0 is a valid empty bitmap.
    let buf = vec![b'F', b'I', 0, 0, 0, 0, 0, 0];
    let v = FrozenBitmapView::from_bytes(&buf).unwrap();
    assert!(v.is_empty() && v.is_inline());
    assert_eq!(v.min(), None);
}
