//! Shared helpers for the op kernel tests.
#![allow(dead_code)]

use std::collections::BTreeSet;

use frostbit::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};
use roaring::RoaringBitmap;

/// Build a frozen bitmap, standard-format or compact, from sorted values.
pub fn build(values: &[u32], standard: bool) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    if standard {
        b.finish_standard()
    } else {
        b.finish()
    }
}

pub fn rb(values: &[u32]) -> RoaringBitmap {
    RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap()
}

pub fn views(bms: &[FrozenBitmap]) -> Vec<FrozenBitmapView<'_>> {
    bms.iter().map(|b| b.view()).collect()
}

pub fn at(k: u16, lo: u16) -> u32 {
    ((k as u32) << 16) | lo as u32
}

pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One container of every encoding, for exhaustive pairings.
pub fn shapes() -> Vec<Vec<u32>> {
    vec![
        (0..300).map(|i| i * 7).collect(),    // array
        (0..2000).collect(),                  // run
        (0..6000).map(|i| i * 2).collect(),   // bitmap
        vec![0, 65_536, 131_072],             // inline-ish
        (0..130).map(|k| at(k, 9)).collect(), // single-per-key
        vec![0, u32::MAX],                    // extremes
    ]
}

/// A random 2..6-way case: each input is a random set over a random key spread,
/// with a random standard/compact build flag per input.
pub fn random_inputs(st: &mut u64) -> (Vec<Vec<u32>>, Vec<bool>) {
    let n = 2 + (splitmix64(st) % 4) as usize;
    let mut inputs: Vec<Vec<u32>> = (0..n)
        .map(|_| {
            let cnt = (splitmix64(st) % 3000) as usize;
            let spread = 1u64 << (17 + (splitmix64(st) % 9));
            let mut s = BTreeSet::new();
            for _ in 0..cnt {
                s.insert((splitmix64(st) % spread) as u32);
            }
            s.into_iter().collect()
        })
        .collect();
    for v in inputs.iter_mut() {
        if v.is_empty() {
            *v = vec![splitmix64(st) as u32];
        }
    }
    let builds: Vec<bool> = (0..n).map(|_| (splitmix64(st) & 1) == 0).collect();
    (inputs, builds)
}
