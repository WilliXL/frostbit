//! Union kernel: value equality vs roaring across shapes/encodings.

use frostbit::ops::kernels::union;

use crate::common::*;

fn check(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<_> = inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let got: Vec<u32> = union(&views(&bms)).view().iter().collect();
    let mut want = rb(&inputs[0]);
    for v in &inputs[1..] {
        want |= rb(v);
    }
    assert_eq!(got, want.iter().collect::<Vec<_>>(), "union mismatch");
}

#[test]
fn empty_and_pairs() {
    assert_eq!(union(&[]).view().len(), 0);
    check(&[vec![1, 2, 3]], &[true]);
    let s = shapes();
    for a in &s {
        for b in &s {
            for &sa in &[true, false] {
                for &sb in &[true, false] {
                    check(&[a.clone(), b.clone()], &[sa, sb]);
                }
            }
        }
    }
}

#[test]
fn promotion_boundaries() {
    // array → bitmap as the merged cardinality crosses the dense threshold.
    let a: Vec<u32> = (0..2500).map(|i| at(0, (i * 2) as u16)).collect();
    let b: Vec<u32> = (0..2500).map(|i| at(0, (i * 2 + 1) as u16)).collect(); // disjoint → bitmap
    check(&[a, b], &[true, true]);

    // run-heavy union crossing MAX_RUNS.
    let mut ra = Vec::new();
    let mut rb_ = Vec::new();
    let mut lo = 0u32;
    for _ in 0..1500 {
        for j in 0..3 {
            ra.push(at(5, (lo + j) as u16));
        }
        lo += 4;
    }
    lo = 2;
    for _ in 0..1500 {
        for j in 0..3 {
            if lo + j <= 0xFFFF {
                rb_.push(at(5, (lo + j) as u16));
            }
        }
        lo += 4;
    }
    rb_.sort_unstable();
    rb_.dedup();
    check(&[ra, rb_], &[true, true]);
}

#[test]
fn three_way() {
    let s = shapes();
    check(&[s[0].clone(), s[1].clone(), s[2].clone()], &[true, false, true]);
}

#[test]
fn ten_million_scale() {
    let a: Vec<u32> = (0..10_000_000u32).step_by(2).collect();
    let b: Vec<u32> = (0..10_000_000u32).step_by(3).collect();
    let bms = [build(&a, true), build(&b, true)];
    let got: Vec<u32> = union(&views(&bms)).view().iter().collect();
    assert_eq!(got, (rb(&a) | rb(&b)).iter().collect::<Vec<_>>());
}

#[test]
fn randomized_differential() {
    let mut st = 0x0D1F_F123_u64;
    for _ in 0..2000 {
        let (inputs, builds) = random_inputs(&mut st);
        check(&inputs, &builds);
    }
}
