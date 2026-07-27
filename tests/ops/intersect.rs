//! Intersection kernel: value equality vs roaring across shapes/encodings.

use frostbit::ops::kernels::intersect_fast as intersect;

use crate::common::*;

fn expect(inputs: &[Vec<u32>]) -> Vec<u32> {
    let mut acc = rb(&inputs[0]);
    for v in &inputs[1..] {
        acc &= rb(v);
    }
    acc.iter().collect()
}

fn check(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<_> = inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let got: Vec<u32> = intersect(&views(&bms)).view().iter().collect();
    assert_eq!(got, expect(inputs), "intersect mismatch");
}

#[test]
fn empty_and_singletons() {
    assert_eq!(intersect(&[]).view().len(), 0);
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
    let bms = [build(&a, true), build(&b, true)];
    let got: Vec<u32> = intersect(&views(&bms)).view().iter().collect();
    assert_eq!(got, (rb(&a) & rb(&b)).iter().collect::<Vec<_>>());
}

#[test]
fn randomized_differential() {
    let mut st = 0x1234_A0D0_u64;
    for _ in 0..2000 {
        let (inputs, builds) = random_inputs(&mut st);
        check(&inputs, &builds);
    }
}
