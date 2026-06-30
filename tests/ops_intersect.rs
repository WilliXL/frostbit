//! Intersection kernel: value equality vs roaring across shapes/encodings.
//! Runs in debug, so the arena's no-alloc `record` debug-assert fires on any
//! slot overflow.
#![cfg(feature = "internals")]

use std::collections::BTreeSet;

use frostbit::ops::kernels::intersect;
use frostbit::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};
use roaring::RoaringBitmap;

fn build(values: &[u32], standard: bool) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    if standard { b.finish_standard() } else { b.finish() }
}

fn rb(values: &[u32]) -> RoaringBitmap {
    RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap()
}

fn expect_intersect(inputs: &[Vec<u32>]) -> Vec<u32> {
    let mut acc = rb(&inputs[0]);
    for v in &inputs[1..] {
        acc &= rb(v);
    }
    acc.iter().collect()
}

fn check(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<FrozenBitmap> =
        inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let views: Vec<FrozenBitmapView<'_>> = bms.iter().map(|b| b.view()).collect();
    let got: Vec<u32> = intersect(&views).view().iter().collect();
    assert_eq!(got, expect_intersect(inputs), "intersect mismatch");
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn at(k: u16, lo: u16) -> u32 {
    ((k as u32) << 16) | lo as u32
}

#[test]
fn empty_and_singletons() {
    let bm = intersect(&[]);
    assert_eq!(bm.view().len(), 0);
    check(&[vec![1, 2, 3]], &[true]); // single input = identity
    check(&[vec![1, 2, 3], vec![]], &[true, true]); // with empty → empty
}

#[test]
fn all_container_pairings() {
    let arr: Vec<u32> = (0..300).map(|i| i * 7).collect();
    let run: Vec<u32> = (0..2000).collect();
    let bmp: Vec<u32> = (0..6000).map(|i| i * 2).collect();
    let inl: Vec<u32> = vec![0, 65_536, 131_072];
    let shapes = [arr, run, bmp, inl];
    for a in &shapes {
        for b in &shapes {
            for &sa in &[true, false] {
                for &sb in &[true, false] {
                    check(&[a.clone(), b.clone()], &[sa, sb]);
                }
            }
        }
    }
}

#[test]
fn overlap_patterns() {
    check(&[vec![1, 2, 3, 4, 5], vec![3, 4, 5, 6, 7]], &[true, true]);
    check(&[vec![1, 2, 3], vec![10, 20, 30]], &[true, true]); // disjoint
    let v: Vec<u32> = (0..1000).collect();
    check(&[v.clone(), v.clone()], &[true, false]); // identical
    check(&[vec![2, 4], (0..100).collect()], &[true, true]); // subset
}

#[test]
fn shared_key_disjoint_values() {
    // Same container key, no overlapping values → empty result, but the plan
    // keeps a slot (over-approx). Exercises the over-approx path.
    let a: Vec<u32> = (0..100).map(|i| at(7, (i * 2) as u16)).collect();
    let b: Vec<u32> = (0..100).map(|i| at(7, (i * 2 + 1) as u16)).collect();
    check(&[a, b], &[true, true]);
}

#[test]
fn dense_bitmap_intersection() {
    // Both dense in the same keys (min_card > 4096 → bitmap accumulator path).
    let a: Vec<u32> = (0..5000).map(|i| at(0, i as u16)).collect();
    let b: Vec<u32> = (1000..6000).map(|i| at(0, i as u16)).collect();
    check(&[a, b], &[true, true]);
}

#[test]
fn three_and_four_way() {
    let a: Vec<u32> = (0..5000).collect();
    let b: Vec<u32> = (0..5000).map(|i| i * 2).collect();
    let c: Vec<u32> = (0..5000).map(|i| i * 3).collect();
    let d: Vec<u32> = (0..10000).collect();
    check(&[a.clone(), b.clone(), c.clone()], &[true, false, true]);
    check(&[a, b, c, d], &[true, true, false, false]);
}

#[test]
fn boundary_cardinalities() {
    for ca in [256u32, 4096, 4097] {
        for cb in [256u32, 4096, 4097] {
            let a: Vec<u32> = (0..ca).map(|i| at(0, (i % 0x1_0000) as u16)).collect();
            let b: Vec<u32> = (0..cb).map(|i| at(0, (i % 0x1_0000) as u16)).collect();
            check(&[a, b], &[true, true]);
        }
    }
}

#[test]
fn ten_million_scale() {
    let a: Vec<u32> = (0..10_000_000u32).step_by(2).collect();
    let b: Vec<u32> = (0..10_000_000u32).step_by(3).collect();
    let av = build(&a, true);
    let bv = build(&b, true);
    let got: Vec<u32> = intersect(&[av.view(), bv.view()]).view().iter().collect();
    let want: Vec<u32> = (rb(&a) & rb(&b)).iter().collect();
    assert_eq!(got, want);
}

#[test]
fn randomized_differential() {
    let mut st = 0x1234_A0D0_u64;
    for _ in 0..2000 {
        let n = 2 + (splitmix64(&mut st) % 4) as usize;
        let mk = |st: &mut u64| -> Vec<u32> {
            let cnt = (splitmix64(st) % 3000) as usize;
            let spread = 1u64 << (17 + (splitmix64(st) % 9));
            let mut s = BTreeSet::new();
            for _ in 0..cnt {
                s.insert((splitmix64(st) % spread) as u32);
            }
            s.into_iter().collect()
        };
        let mut inputs: Vec<Vec<u32>> = (0..n).map(|_| mk(&mut st)).collect();
        for v in inputs.iter_mut() {
            if v.is_empty() {
                *v = vec![splitmix64(&mut st) as u32];
            }
        }
        let builds: Vec<bool> = (0..n).map(|_| (splitmix64(&mut st) & 1) == 0).collect();
        check(&inputs, &builds);
    }
}
