//! Wire-format byte primitives: little-endian round-trips, byte order, and
//! alignment math. Requires the `internals` feature (run with
//! `cargo test --features internals`).
#![cfg(feature = "internals")]

use frostbit::format::*;

#[test]
fn u16_roundtrip_and_byte_order() {
    let mut buf = [0u8; 8];
    write_u16(&mut buf, 0, 0x1234);
    assert_eq!(read_u16(&buf, 0), 0x1234);
    // Little-endian on the wire.
    assert_eq!(buf[0], 0x34);
    assert_eq!(buf[1], 0x12);
}

#[test]
fn u32_roundtrip_and_byte_order() {
    let mut buf = [0u8; 8];
    write_u32(&mut buf, 0, 0x1122_3344);
    assert_eq!(read_u32(&buf, 0), 0x1122_3344);
    assert_eq!(&buf[0..4], &[0x44, 0x33, 0x22, 0x11]);
}

#[test]
fn u64_roundtrip_and_byte_order() {
    let mut buf = [0u8; 16];
    write_u64(&mut buf, 0, 0x0102_0304_0506_0708);
    assert_eq!(read_u64(&buf, 0), 0x0102_0304_0506_0708);
    assert_eq!(&buf[0..8], &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
}

#[test]
fn read_write_at_nonzero_offset() {
    let mut buf = [0u8; 32];
    write_u16(&mut buf, 3, 0xBEEF);
    write_u32(&mut buf, 5, 0xDEAD_C0DE);
    write_u64(&mut buf, 9, 0xFEED_FACE_CAFE_BABE);
    assert_eq!(read_u16(&buf, 3), 0xBEEF);
    assert_eq!(read_u32(&buf, 5), 0xDEAD_C0DE);
    assert_eq!(read_u64(&buf, 9), 0xFEED_FACE_CAFE_BABE);
    // Bytes outside the written ranges are untouched.
    assert_eq!(buf[0..3], [0, 0, 0]);
    assert_eq!(buf[17..], [0u8; 15]);
}

#[test]
fn extremes_roundtrip() {
    let mut buf = [0u8; 8];
    write_u16(&mut buf, 0, u16::MAX);
    assert_eq!(read_u16(&buf, 0), u16::MAX);
    write_u32(&mut buf, 0, u32::MAX);
    assert_eq!(read_u32(&buf, 0), u32::MAX);
    write_u64(&mut buf, 0, u64::MAX);
    assert_eq!(read_u64(&buf, 0), u64::MAX);
}

#[test]
fn align_up_to_8() {
    assert_eq!(align_up(0, 8), 0);
    assert_eq!(align_up(1, 8), 8);
    assert_eq!(align_up(7, 8), 8);
    assert_eq!(align_up(8, 8), 8);
    assert_eq!(align_up(9, 8), 16);
}

#[test]
fn align_up_to_64() {
    assert_eq!(align_up(0, 64), 0);
    assert_eq!(align_up(1, 64), 64);
    assert_eq!(align_up(13, 64), 64);
    assert_eq!(align_up(64, 64), 64);
    assert_eq!(align_up(65, 64), 128);
    // Header (16) rounded to the bitmap-container alignment.
    assert_eq!(align_up(HEADER_SIZE, BUF_ALIGN), 64);
}

#[test]
fn wire_constants_are_consistent() {
    assert_eq!(HEADER_SIZE, 16);
    assert_eq!(BITMAP_WORDS, 1024);
    assert_eq!(BITMAP_BYTES, 8192);
    assert_eq!(ARRAY_MAX_SIZE, 4096);
    assert_eq!(BUF_ALIGN, 64);
    assert_eq!(VERSION, 3);
    // The magic reads "FROZ" in big-endian hex (matches the jata reference).
    assert_eq!(MAGIC, 0x4652_4F5A);
    assert_eq!(MAGIC.to_be_bytes(), *b"FROZ");
    // Flags are distinct single bits.
    assert_ne!(FLAG_HAS_RUNS, FLAG_FULL);
    assert_eq!(FLAG_HAS_RUNS & FLAG_FULL, 0);
}

// ---------------------------------------------------------------------------
// Header codec
// ---------------------------------------------------------------------------

fn write_header(h: &Header) -> [u8; HEADER_SIZE] {
    let mut buf = [0u8; HEADER_SIZE];
    h.write(&mut buf);
    buf
}

#[test]
fn header_roundtrips() {
    let cases = [
        Header { has_runs: false, has_bitmap: false, num_containers: 0, cardinality: 0 },
        Header { has_runs: false, has_bitmap: false, num_containers: 1, cardinality: 1 },
        Header { has_runs: true, has_bitmap: true, num_containers: 153, cardinality: 10_000_000 },
        Header { has_runs: false, has_bitmap: true, num_containers: 65_536, cardinality: 4_294_967_295 },
    ];
    for h in cases {
        let buf = write_header(&h);
        assert_eq!(Header::parse(&buf), Some(h), "roundtrip failed for {h:?}");
    }
}

#[test]
fn header_full_bitmap_uses_flag() {
    // 2^32 can't fit the u32 field; FLAG_FULL must carry it.
    let h = Header { has_runs: false, has_bitmap: true, num_containers: 65_536, cardinality: FULL_CARDINALITY };
    let buf = write_header(&h);
    assert_ne!(read_u16(&buf, H_FLAGS) & FLAG_FULL, 0, "FLAG_FULL not set");
    assert_eq!(read_u32(&buf, H_CARDINALITY), 0, "card field should be 0 when full");
    assert_eq!(Header::parse(&buf).unwrap().cardinality, FULL_CARDINALITY);
}

#[test]
fn header_flags_are_independent() {
    let h = Header { has_runs: true, has_bitmap: false, num_containers: 3, cardinality: FULL_CARDINALITY };
    let buf = write_header(&h);
    let parsed = Header::parse(&buf).unwrap();
    assert!(parsed.has_runs);
    assert!(!parsed.has_bitmap);
    assert_eq!(parsed.cardinality, FULL_CARDINALITY);
}

#[test]
fn header_rejects_garbage() {
    let h = Header { has_runs: false, has_bitmap: false, num_containers: 1, cardinality: 1 };
    // Too short.
    assert_eq!(Header::parse(&[0u8; HEADER_SIZE - 1]), None);
    // Bad magic.
    let mut buf = write_header(&h);
    buf[0] ^= 0xFF;
    assert_eq!(Header::parse(&buf), None);
    // Bad version.
    let mut buf = write_header(&h);
    write_u16(&mut buf, H_VERSION, 99);
    assert_eq!(Header::parse(&buf), None);
}

#[test]
fn header_only_touches_its_16_bytes() {
    let mut buf = [0xAAu8; 32];
    Header { has_runs: false, has_bitmap: false, num_containers: 7, cardinality: 7 }.write(&mut buf);
    // Trailing bytes are left for the index/data sections.
    assert_eq!(buf[HEADER_SIZE..], [0xAAu8; 16]);
}

#[test]
fn data_section_off_alignment() {
    // No bitmap: 8-aligned just past the index.
    assert_eq!(data_section_off(0, false), 16);
    assert_eq!(data_section_off(1, false), 24); // 16 + 8
    assert_eq!(data_section_off(3, false), 40); // 16 + 24
    // With a bitmap: bumped to the next 64 boundary.
    assert_eq!(data_section_off(0, true), 64);
    assert_eq!(data_section_off(1, true), 64);
    assert_eq!(data_section_off(6, true), 64); // 16 + 48 = 64
    assert_eq!(data_section_off(7, true), 128); // 16 + 56 = 72 -> 128
    for n in 0..100 {
        assert_eq!(data_section_off(n, true) % 64, 0);
        assert_eq!(data_section_off(n, false) % 8, 0);
    }
}

// ---------------------------------------------------------------------------
// SoA container index
// ---------------------------------------------------------------------------

fn sample_entries() -> Vec<IndexEntry> {
    vec![
        IndexEntry { key: 0, typ: CT_ARRAY, cardinality: 1, data_offset: 0 },
        IndexEntry { key: 5, typ: CT_BITMAP, cardinality: 65_536, data_offset: 8 },
        IndexEntry { key: 200, typ: CT_RUN, cardinality: 1000, data_offset: 8200 },
        IndexEntry { key: 65_535, typ: CT_ARRAY, cardinality: 42, data_offset: E_OFFSET_MASK },
    ]
}

/// A `WORD_ALIGN`-aligned index buffer. `Index::new` requires the alignment
/// both `from_bytes` boundaries enforce; a bare `Vec<u8>` only appears to
/// satisfy it because allocators over-align — under Miri it does not.
fn build_index(entries: &[IndexEntry]) -> aligned_vec::AVec<u8> {
    let n = entries.len();
    let mut buf = aligned_vec::AVec::new(WORD_ALIGN);
    buf.resize(HEADER_SIZE + index_size(n), 0u8);
    for (i, e) in entries.iter().enumerate() {
        write_index_entry(&mut buf, n, i, *e);
    }
    buf
}

#[test]
fn index_entry_roundtrips() {
    let entries = sample_entries();
    let n = entries.len();
    let buf = build_index(&entries);
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(read_index_entry(&buf, n, i), *e, "entry {i} mismatch");
    }
}

#[test]
fn index_is_structure_of_arrays() {
    // Keys must be physically contiguous right after the header so a search
    // touches only them.
    let entries = sample_entries();
    let n = entries.len();
    let buf = build_index(&entries);
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(read_u16(&buf, keys_off() + i * 2), e.key);
    }
    assert_eq!(keys_off(), HEADER_SIZE);
    assert_eq!(cards_off(n), HEADER_SIZE + 2 * n);
    assert_eq!(type_offsets_off(n), HEADER_SIZE + 4 * n);
    // type_offsets sub-array is 4-byte aligned for zero-copy u32 access.
    assert_eq!(type_offsets_off(n) % 4, 0);
    assert_eq!(index_size(n), 8 * n);
}

#[test]
fn type_and_offset_packing_independent() {
    // Max offset coexists with each container type without clobbering.
    for typ in [CT_ARRAY, CT_BITMAP, CT_RUN] {
        let e = IndexEntry { key: 1, typ, cardinality: 1, data_offset: E_OFFSET_MASK };
        let buf = build_index(&[e]);
        assert_eq!(read_index_entry(&buf, 1, 0), e);
    }
}

#[test]
fn find_container_hits_and_misses() {
    let entries = sample_entries(); // keys: 0, 5, 200, 65535
    let n = entries.len();
    let buf = build_index(&entries);
    assert_eq!(Index::new(&buf, n).find(0), Some(0));
    assert_eq!(Index::new(&buf, n).find(5), Some(1));
    assert_eq!(Index::new(&buf, n).find(200), Some(2));
    assert_eq!(Index::new(&buf, n).find(65_535), Some(3));
    for missing in [1u16, 4, 6, 199, 201, 65_534] {
        assert_eq!(Index::new(&buf, n).find(missing), None, "found absent key {missing}");
    }
    // Empty index.
    assert_eq!(Index::new(&buf, 0).find(0), None);
}
