//! Difference kernel (`inputs[0]` minus the rest): value equality vs roaring.

use frostbit::ops::kernels::diff;

use crate::common::*;

fn check(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<_> = inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let got: Vec<u32> = diff(&views(&bms)).view().iter().collect();
    let mut want = rb(&inputs[0]);
    for v in &inputs[1..] {
        want -= rb(v);
    }
    assert_eq!(got, want.iter().collect::<Vec<_>>(), "diff mismatch");
}

#[test]
fn empty_and_pairs() {
    assert_eq!(diff(&[]).view().len(), 0);
    check(&[vec![1, 2, 3]], &[true]); // no rhs → verbatim A
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
fn dense_bitmap_path() {
    // A dense (card > 4096) so the bitmap accumulator path is taken.
    let a: Vec<u32> = (0..6000).map(|i| at(0, i as u16)).collect();
    let b: Vec<u32> = (2000..4000).map(|i| at(0, i as u16)).collect();
    check(&[a, b], &[true, true]);
}

#[test]
fn disjoint_keys_verbatim() {
    // B touches none of A's keys → every A container copied verbatim.
    let a: Vec<u32> = (0..5).flat_map(|k| (0..1000).map(move |i| at(k, i))).collect();
    let b: Vec<u32> = (100..105).map(|k| at(k, 1)).collect();
    check(&[a, b], &[true, true]);
}

#[test]
fn three_way() {
    let s = shapes();
    check(&[s[2].clone(), s[0].clone(), s[1].clone()], &[true, true, false]);
}

#[test]
fn ten_million_scale() {
    let a: Vec<u32> = (0..10_000_000u32).step_by(2).collect();
    let b: Vec<u32> = (0..10_000_000u32).step_by(3).collect();
    let bms = [build(&a, true), build(&b, true)];
    let got: Vec<u32> = diff(&views(&bms)).view().iter().collect();
    assert_eq!(got, (rb(&a) - rb(&b)).iter().collect::<Vec<_>>());
}

#[test]
fn randomized_differential() {
    let mut st = 0x0D1F_F123_u64;
    for _ in 0..2000 {
        let (inputs, builds) = random_inputs(&mut st);
        check(&inputs, &builds);
    }
}
