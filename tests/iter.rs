//! `iter()`: exact ascending round-trip across encodings and container types.

use frostbit::{FrozenBitmap, FrozenBitmapBuilder};

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

fn check(values: &[u32]) {
    let bm = build(values);
    let got: Vec<u32> = bm.iter().collect();
    assert_eq!(got, values, "{} values", values.len());
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn per_shape_roundtrips() {
    check(&[]);
    check(&[42]); // inline
    check(&[0, 65_536, 131_072, u32::MAX]); // inline, multi-key
    check(&(0..100).map(|i| i * 3).collect::<Vec<_>>()); // array
    check(&(0..1000).collect::<Vec<_>>()); // run
    check(&(0..5000).map(|i| i * 2).collect::<Vec<_>>()); // bitmap
}

#[test]
fn run_ending_at_container_edge() {
    // Run end 0xFFFF must hand off to the next container without wrapping.
    let mut vals: Vec<u32> = (0xFF00..=0xFFFF).collect();
    vals.extend(0x1_0000..0x1_0010);
    check(&vals);
}

#[test]
fn multi_container_mixed_types() {
    let mut vals: Vec<u32> = (0..1000).collect(); // run
    vals.extend((131_072..131_372).map(|v| v)); // run
    vals.extend((262_144..262_344).map(|i| i * 2 - 262_144)); // array-ish spread
    vals.sort_unstable();
    vals.dedup();
    check(&vals);
}

#[test]
fn size_hint_tracks_remaining() {
    let vals: Vec<u32> = (0..500).map(|i| i * 7).collect();
    let bm = build(&vals);
    let mut it = bm.iter();
    assert_eq!(it.size_hint(), (500, Some(500)));
    for consumed in 1..=100 {
        it.next().unwrap();
        assert_eq!(it.size_hint(), (500 - consumed, Some(500 - consumed)));
    }
}

#[test]
fn fused_after_exhaustion() {
    let bm = build(&[1, 2, 3]);
    let mut it = bm.iter();
    assert_eq!(it.by_ref().count(), 3);
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
}

#[test]
fn large_differential() {
    // ~1.07M values: dense run block + even bitmap block + pseudo-random scatter.
    let mut set: std::collections::BTreeSet<u32> = (0..1_000_000).collect();
    set.extend((2_000_000..2_100_000).step_by(2));
    let mut st = 0xF005_u64;
    for _ in 0..30_000 {
        set.insert((splitmix64(&mut st) % (1 << 30)) as u32);
    }
    let vals: Vec<u32> = set.iter().copied().collect();
    let bm = build(&vals);
    assert_eq!(bm.view().len(), vals.len() as u64);
    let got: Vec<u32> = bm.iter().collect();
    assert_eq!(got, vals);
}
