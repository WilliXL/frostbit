//! Roaring ↔ frozen round-trips, including the 10M-element acceptance tests.
//! Invariants per distribution:
//!   roaring → frozen → roaring is identity (set equality),
//!   frozen → roaring → frozen is identity (byte equality — compact is
//!   deterministic in the value set), and contains() agrees on probes.
#![cfg(feature = "roaring")]

use frostbit::FrozenBitmap;
use roaring::RoaringBitmap;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn assert_roundtrips(rb: &RoaringBitmap) {
    let fz = FrozenBitmap::from_roaring(rb);
    let v = fz.view();
    assert_eq!(v.len(), rb.len());
    assert_eq!(v.min(), rb.min());
    assert_eq!(v.max(), rb.max());

    let back = fz.to_roaring();
    assert_eq!(&back, rb, "roaring -> frozen -> roaring mismatch");
    assert_eq!(
        FrozenBitmap::from_roaring(&back),
        fz,
        "frozen -> roaring -> frozen not byte-identical"
    );
}

fn assert_probes(rb: &RoaringBitmap, fz: &FrozenBitmap, probes: impl Iterator<Item = u32>) {
    let v = fz.view();
    for p in probes {
        assert_eq!(v.contains(p), rb.contains(p), "probe {p}");
    }
}

#[test]
fn small_shapes() {
    let shapes: Vec<Vec<u32>> = vec![
        vec![],
        vec![42],
        vec![0, 65_536, 131_072, u32::MAX],            // inline
        (0..100).map(|i| i * 3).collect(),             // array
        (0..1000).collect(),                           // run
        (0..5000).map(|i| i * 2).collect(),            // bitmap
        (0..65_536).collect(),                         // one full container
    ];
    for vals in shapes {
        let rb: RoaringBitmap = vals.iter().copied().collect();
        assert_roundtrips(&rb);
    }
}

#[test]
fn ten_million_dense() {
    // 0..10M: ~153 run containers.
    let mut rb = RoaringBitmap::new();
    rb.insert_range(0..10_000_000);
    assert_roundtrips(&rb);

    let fz = FrozenBitmap::from_roaring(&rb);
    assert!(fz.view().num_containers() > 100, "expected many containers");
    assert_probes(&rb, &fz, (0..12_000_000u32).step_by(100_007));
    // Dense bitmaps compress to runs: ~6 bytes/container.
    assert!(fz.byte_len() < 16_384, "dense should be tiny, got {}", fz.byte_len());
}

#[test]
fn ten_million_even_spread() {
    // Evens over 0..20M: ~305 bitmap containers.
    let rb = RoaringBitmap::from_sorted_iter((0..10_000_000u32).map(|i| i * 2)).unwrap();
    assert_roundtrips(&rb);

    let fz = FrozenBitmap::from_roaring(&rb);
    assert!(fz.view().num_containers() > 300);
    assert_probes(&rb, &fz, (0..20_000_000u32).step_by(99_991));
}

#[test]
fn ten_million_sparse_random() {
    // ~10M pseudo-random values over the full u32 space: arrays in every key.
    let mut st = 0x10_000_000_u64;
    let mut vals: Vec<u32> = (0..10_000_000).map(|_| splitmix64(&mut st) as u32).collect();
    vals.sort_unstable();
    vals.dedup();
    let rb = RoaringBitmap::from_sorted_iter(vals.iter().copied()).unwrap();
    assert_eq!(rb.len(), vals.len() as u64);
    assert_roundtrips(&rb);

    let fz = FrozenBitmap::from_roaring(&rb);
    assert_eq!(fz.view().num_containers(), 65_536, "scatter should hit every key");
    let mut st = 0xBEEF_u64;
    assert_probes(&rb, &fz, (0..100_000).map(|_| splitmix64(&mut st) as u32));
}

#[test]
fn ten_million_mixed() {
    // Dense run block + even bitmap block + wide scatter.
    let mut rb = RoaringBitmap::new();
    rb.insert_range(0..3_000_000);
    let evens = RoaringBitmap::from_sorted_iter((4_000_000..10_000_000u32).filter(|v| v % 2 == 0))
        .unwrap();
    rb |= evens;
    let mut st = 0x0003_1337_u64;
    for _ in 0..1_000_000 {
        rb.insert(100_000_000 + (splitmix64(&mut st) % 900_000_000) as u32);
    }
    assert!(rb.len() > 6_900_000);
    assert_roundtrips(&rb);

    let fz = FrozenBitmap::from_roaring(&rb);
    let mut st = 0xD00D_u64;
    assert_probes(&rb, &fz, (0..100_000).map(|_| splitmix64(&mut st) % 1_100_000_000) .map(|v| v as u32));
}
