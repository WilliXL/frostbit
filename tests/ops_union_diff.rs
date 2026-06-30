//! Union and difference kernels: value equality vs roaring. Debug build, so
//! the arena's no-alloc `record` debug-assert fires on any slot overflow.
#![cfg(feature = "internals")]

use std::collections::BTreeSet;

use frostbit::ops::kernels::{diff, union};
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

fn views<'a>(bms: &'a [FrozenBitmap]) -> Vec<FrozenBitmapView<'a>> {
    bms.iter().map(|b| b.view()).collect()
}

fn check_union(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<FrozenBitmap> =
        inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let got: Vec<u32> = union(&views(&bms)).view().iter().collect();
    let mut want = rb(&inputs[0]);
    for v in &inputs[1..] {
        want |= rb(v);
    }
    assert_eq!(got, want.iter().collect::<Vec<_>>(), "union mismatch");
}

fn check_diff(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<FrozenBitmap> =
        inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let got: Vec<u32> = diff(&views(&bms)).view().iter().collect();
    let mut want = rb(&inputs[0]);
    for v in &inputs[1..] {
        want -= rb(v);
    }
    assert_eq!(got, want.iter().collect::<Vec<_>>(), "diff mismatch");
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

fn shapes() -> Vec<Vec<u32>> {
    vec![
        (0..300).map(|i| i * 7).collect(),         // array
        (0..2000).collect(),                       // run
        (0..6000).map(|i| i * 2).collect(),        // bitmap
        vec![0, 65_536, 131_072],                  // inline-ish
        (0..130).map(|k| at(k, 9)).collect(),      // single-per-key
        vec![0, u32::MAX],                         // extremes
    ]
}

#[test]
fn union_empty_and_pairs() {
    assert_eq!(union(&[]).view().len(), 0);
    check_union(&[vec![1, 2, 3]], &[true]);
    let s = shapes();
    for a in &s {
        for b in &s {
            for &sa in &[true, false] {
                for &sb in &[true, false] {
                    check_union(&[a.clone(), b.clone()], &[sa, sb]);
                }
            }
        }
    }
}

#[test]
fn diff_empty_and_pairs() {
    assert_eq!(diff(&[]).view().len(), 0);
    check_diff(&[vec![1, 2, 3]], &[true]); // no rhs → verbatim A
    let s = shapes();
    for a in &s {
        for b in &s {
            for &sa in &[true, false] {
                for &sb in &[true, false] {
                    check_diff(&[a.clone(), b.clone()], &[sa, sb]);
                }
            }
        }
    }
}

#[test]
fn union_promotion_boundaries() {
    // array→bitmap as the merged cardinality crosses 4096 within a key.
    let a: Vec<u32> = (0..2500).map(|i| at(0, (i * 2) as u16)).collect();
    let b: Vec<u32> = (0..2500).map(|i| at(0, (i * 2 + 1) as u16)).collect(); // disjoint, sum 5000 → bitmap
    check_union(&[a, b], &[true, true]);
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
    check_union(&[ra, rb_], &[true, true]);
}

#[test]
fn diff_dense_bitmap_path() {
    // A dense (card > 4096) so the bitmap accumulator path is taken.
    let a: Vec<u32> = (0..6000).map(|i| at(0, i as u16)).collect();
    let b: Vec<u32> = (2000..4000).map(|i| at(0, i as u16)).collect();
    check_diff(&[a, b], &[true, true]);
}

#[test]
fn diff_disjoint_keys_verbatim() {
    // B touches none of A's keys → every A container copied verbatim.
    let a: Vec<u32> = (0..5).flat_map(|k| (0..1000).map(move |i| at(k, i))).collect();
    let b: Vec<u32> = (100..105).map(|k| at(k, 1)).collect();
    check_diff(&[a, b], &[true, true]);
}

#[test]
fn three_way() {
    let s = shapes();
    check_union(&[s[0].clone(), s[1].clone(), s[2].clone()], &[true, false, true]);
    check_diff(&[s[2].clone(), s[0].clone(), s[1].clone()], &[true, true, false]);
}

#[test]
fn ten_million_scale() {
    let a: Vec<u32> = (0..10_000_000u32).step_by(2).collect();
    let b: Vec<u32> = (0..10_000_000u32).step_by(3).collect();
    let bms = [build(&a, true), build(&b, true)];
    let v = views(&bms);
    assert_eq!(
        union(&v).view().iter().collect::<Vec<_>>(),
        (rb(&a) | rb(&b)).iter().collect::<Vec<_>>()
    );
    assert_eq!(
        diff(&v).view().iter().collect::<Vec<_>>(),
        (rb(&a) - rb(&b)).iter().collect::<Vec<_>>()
    );
}

#[test]
fn randomized_differential() {
    let mut st = 0x0D1F_F123_u64;
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
        check_union(&inputs, &builds);
        check_diff(&inputs, &builds);
    }
}
