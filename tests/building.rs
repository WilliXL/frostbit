//! Builder: standard wire structure (via `finish_standard`) and compact
//! encoding selection (via `finish`).
#![cfg(feature = "internals")]

use frostbit::format::*;
use frostbit::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};

fn build_std(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish_standard()
}

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

fn header(bm: &FrozenBitmap) -> Header {
    Header::parse(bm.as_bytes()).expect("valid header")
}

fn entries(bm: &FrozenBitmap) -> Vec<IndexEntry> {
    let h = header(bm);
    let n = h.num_containers as usize;
    (0..n).map(|i| read_index_entry(bm.as_bytes(), n, i)).collect()
}

/// Read an array container's lows back out of the data section.
fn array_lows(bm: &FrozenBitmap, e: &IndexEntry, n: usize) -> Vec<u16> {
    assert_eq!(e.typ, CT_ARRAY);
    let base = data_section_off(n, header(bm).has_bitmap) + e.data_offset as usize;
    (0..e.cardinality as usize)
        .map(|j| read_u16(bm.as_bytes(), base + j * 2))
        .collect()
}

// --- Standard structure (finish_standard) -----------------------------------

#[test]
fn empty_standard() {
    let bm = build_std(&[]);
    let h = header(&bm);
    assert_eq!(h.num_containers, 0);
    assert_eq!(h.cardinality, 0);
    assert!(!h.has_runs && !h.has_bitmap);
    assert_eq!(bm.byte_len(), HEADER_SIZE);
}

#[test]
fn single_value_is_array() {
    let bm = build_std(&[42]);
    let es = entries(&bm);
    assert_eq!(es.len(), 1);
    assert_eq!(es[0].key, 0);
    assert_eq!(es[0].typ, CT_ARRAY);
    assert_eq!(es[0].cardinality, 1);
    assert_eq!(array_lows(&bm, &es[0], 1), vec![42]);
}

#[test]
fn sparse_one_key_is_array() {
    let vals: Vec<u32> = (0..100).map(|i| i * 3).collect();
    let bm = build_std(&vals);
    let es = entries(&bm);
    assert_eq!(es.len(), 1);
    assert_eq!(es[0].typ, CT_ARRAY);
    assert_eq!(es[0].cardinality, 100);
    let expected: Vec<u16> = vals.iter().map(|&v| v as u16).collect();
    assert_eq!(array_lows(&bm, &es[0], 1), expected);
    assert!(!header(&bm).has_bitmap);
}

#[test]
fn dense_scattered_is_bitmap_and_64_aligned() {
    // 5000 even values in one key: array=10000B, bitmap=8192B, runs=5000 → bitmap wins.
    let vals: Vec<u32> = (0..5000).map(|i| i * 2).collect();
    let bm = build_std(&vals);
    let h = header(&bm);
    assert!(h.has_bitmap);
    let es = entries(&bm);
    assert_eq!(es[0].typ, CT_BITMAP);
    assert_eq!(es[0].cardinality, 5000);
    let base = data_section_off(1, true) + es[0].data_offset as usize;
    let addr = bm.as_bytes().as_ptr() as usize + base;
    assert_eq!(addr % 64, 0, "bitmap payload not 64-aligned");
}

#[test]
fn consecutive_is_run() {
    let bm = build_std(&(0..1000).collect::<Vec<_>>());
    let h = header(&bm);
    assert!(h.has_runs && !h.has_bitmap);
    let es = entries(&bm);
    assert_eq!(es[0].typ, CT_RUN);
    assert_eq!(es[0].cardinality, 1000);
    // One run: [num_runs=1][start=0][length=999].
    let base = data_section_off(1, false) + es[0].data_offset as usize;
    assert_eq!(read_u16(bm.as_bytes(), base), 1);
    assert_eq!(read_u16(bm.as_bytes(), base + 2), 0);
    assert_eq!(read_u16(bm.as_bytes(), base + 4), 999);
}

#[test]
fn multi_container_keys_ascending() {
    let mut vals = Vec::new();
    for base in [0u32, 65_536, 131_072] {
        for i in 0..50 {
            vals.push(base + i * 10);
        }
    }
    let bm = build_std(&vals);
    let h = header(&bm);
    assert_eq!(h.num_containers, 3);
    assert_eq!(h.cardinality, 150);
    let es = entries(&bm);
    assert_eq!(es.iter().map(|e| e.key).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert!(es.windows(2).all(|w| w[0].key < w[1].key));
}

#[test]
fn mixed_container_types_in_one_bitmap() {
    let mut vals = Vec::new();
    vals.extend(0..1000); // key 0: run
    vals.extend((0..100).map(|i| 131_072 + i * 5)); // key 2: array
    vals.extend((196_608..196_608 + 10_000).step_by(2)); // key 3: bitmap
    let bm = build_std(&vals);
    let h = header(&bm);
    assert!(h.has_runs && h.has_bitmap);
    let es = entries(&bm);
    let types: Vec<u8> = es.iter().map(|e| e.typ).collect();
    assert!(types.contains(&CT_RUN));
    assert!(types.contains(&CT_ARRAY));
    assert!(types.contains(&CT_BITMAP));
    for e in &es {
        if e.typ == CT_BITMAP {
            let base = data_section_off(es.len(), true) + e.data_offset as usize;
            let addr = bm.as_bytes().as_ptr() as usize + base;
            assert_eq!(addr % 64, 0);
        }
    }
}

#[test]
#[should_panic(expected = "ascending")]
fn rejects_descending() {
    let mut b = FrozenBitmapBuilder::new();
    b.push(10);
    b.push(5);
}

#[test]
#[should_panic(expected = "ascending")]
fn rejects_duplicate() {
    let mut b = FrozenBitmapBuilder::new();
    b.push(10);
    b.push(10);
}

// --- Compact selection (finish) ----------------------------------------------

#[test]
fn compact_empty_is_inline() {
    let bm = build(&[]);
    assert!(has_inline_magic(bm.as_bytes()));
    assert_eq!(bm.byte_len(), inline_size(0));
    assert_eq!(read_u16(bm.as_bytes(), INLINE_COUNT_OFF), 0);
}

#[test]
fn compact_scattered_picks_inline() {
    let vals = [0u32, 65_536, 131_072, 196_608];
    let bm = build(&vals);
    assert!(has_inline_magic(bm.as_bytes()));
    assert_eq!(bm.byte_len(), inline_size(4));
    assert_eq!(read_u16(bm.as_bytes(), INLINE_COUNT_OFF), 4);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(read_u32(bm.as_bytes(), INLINE_HEADER_SIZE + i * 4), v);
    }
}

#[test]
fn compact_clustered_picks_standard() {
    // One run container: standard (~24B) beats inline (404B).
    let bm = build(&(0..100).collect::<Vec<_>>());
    assert!(!has_inline_magic(bm.as_bytes()));
    assert!(Header::parse(bm.as_bytes()).is_some());

    // Sparse single key: array payload still beats inline.
    let bm = build(&(0..100).map(|i| i * 3).collect::<Vec<_>>());
    assert!(!has_inline_magic(bm.as_bytes()));
}

#[test]
fn compact_inline_beats_v2_u8_cap() {
    // 1000 single-value containers: inline (~4KB) crushes standard (~10KB).
    // v2's u8 count capped inline at 127 values; the u16 count allows this.
    let vals: Vec<u32> = (0..1000u32).map(|i| i << 16).collect();
    let bm = build(&vals);
    assert!(has_inline_magic(bm.as_bytes()));
    assert_eq!(bm.byte_len(), inline_size(1000));
    let v = FrozenBitmapView::from_bytes(bm.as_bytes()).unwrap();
    assert_eq!(v.len(), 1000);
    assert_eq!(v.min(), Some(0));
    assert_eq!(v.max(), Some(999 << 16));
}

#[test]
fn compact_inline_count_capped_at_u16() {
    // 65536 values can't fit a u16 count → standard despite the scatter.
    let mut b = FrozenBitmapBuilder::new();
    for i in 0..65_536u32 {
        b.push(i << 16);
    }
    let bm = b.finish();
    assert!(!has_inline_magic(bm.as_bytes()));
    assert_eq!(Header::parse(bm.as_bytes()).unwrap().num_containers, 65_536);
}

#[test]
fn compact_inline_is_8_padded() {
    for n in 1usize..=6 {
        let vals: Vec<u32> = (0..n as u32).map(|i| i << 16).collect();
        let bm = build(&vals);
        assert!(has_inline_magic(bm.as_bytes()));
        assert_eq!(bm.byte_len(), inline_size(n));
        assert_eq!(bm.byte_len() % 8, 0);
    }
}

#[test]
fn compact_inline_expands_all_container_types() {
    // Mixed run + array containers, scattered enough that inline wins.
    let vals = [0u32, 1, 2, 3, 65_536, 131_072, 196_613, 262_144];
    let bm = build(&vals);
    assert!(has_inline_magic(bm.as_bytes()));
    let v = FrozenBitmapView::from_bytes(bm.as_bytes()).unwrap();
    assert_eq!(v.len(), vals.len() as u64);
    assert_eq!(v.min(), Some(0));
    assert_eq!(v.max(), Some(262_144));
}
