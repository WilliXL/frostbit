//! `contains()` across all encodings/container types, differential vs BTreeSet.

use std::collections::BTreeSet;

use frostbit::{FrozenBitmap, FrozenBitmapBuilder};

mod support;
use support::splitmix64;

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}


/// Members hit, neighbors checked both ways.
fn check_members_and_neighbors(values: &[u32]) {
    let set: BTreeSet<u32> = values.iter().copied().collect();
    let bm = build(values);
    let v = bm.view();
    for &x in values {
        assert!(v.contains(x), "missing member {x}");
        for probe in [x.wrapping_sub(1), x.wrapping_add(1)] {
            assert_eq!(v.contains(probe), set.contains(&probe), "neighbor {probe}");
        }
    }
}

#[test]
fn empty_contains_nothing() {
    let v = build(&[]);
    let v = v.view();
    for probe in [0, 1, 0xFFFF, 0x10000, u32::MAX] {
        assert!(!v.contains(probe));
    }
}

#[test]
fn inline_membership() {
    check_members_and_neighbors(&[0, 65_536, 131_072, 196_608, u32::MAX]);
    check_members_and_neighbors(&[42]);
}

#[test]
fn array_membership() {
    check_members_and_neighbors(&(0..100).map(|i| i * 3).collect::<Vec<_>>());
}

#[test]
fn run_membership() {
    let mut vals: Vec<u32> = (100..200).collect();
    vals.extend(500..600);
    vals.extend(0xFFF0..=0xFFFF); // run ending at the container edge
    check_members_and_neighbors(&vals);
}

#[test]
fn bitmap_membership() {
    check_members_and_neighbors(&(0..5000).map(|i| i * 2).collect::<Vec<_>>());
}

#[test]
fn container_boundaries() {
    check_members_and_neighbors(&[0xFFFE, 0xFFFF, 0x1_0000, 0x1_0001]);
    check_members_and_neighbors(&[0, u32::MAX]);
}

#[test]
fn differential_mixed_shapes() {
    // Dense run block, even-spread bitmap block, pseudo-random array scatter.
    let mut vals: BTreeSet<u32> = (0..70_000).collect();
    vals.extend((200_000..330_000).step_by(2));
    let mut st = 0x5EED_u64;
    while vals.len() < 350_000 {
        vals.insert((splitmix64(&mut st) % (1 << 27)) as u32 + (1 << 22));
    }
    let sorted: Vec<u32> = vals.iter().copied().collect();
    let bm = build(&sorted);
    let v = bm.view();

    // Every 97th member.
    for &x in sorted.iter().step_by(97) {
        assert!(v.contains(x));
    }
    // 200k random probes.
    let mut st = 0xCAFE_u64;
    for _ in 0..200_000 {
        let probe = (splitmix64(&mut st) % (1 << 28)) as u32;
        assert_eq!(v.contains(probe), vals.contains(&probe), "probe {probe}");
    }
}
