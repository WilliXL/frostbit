//! Shared benchmark helpers: deterministic inputs and matched frozen/roaring
//! sets, so every bench compares the two engines on identical data.
//!
//! CAVEAT — fairness: `roaring`'s `simd` feature is nightly-only
//! (`#![feature(portable_simd)]`) and off by default, so on a stable toolchain
//! roaring runs **scalar** while frostbit uses hand-written SIMD. These numbers
//! therefore flatter frostbit on bitmap-dense work; for a SIMD-vs-SIMD
//! comparison, build on nightly with `roaring`'s `simd` feature enabled.
#![allow(dead_code)]

use frostbit::{FrozenBitmap, FrozenBitmapView};
use roaring::RoaringBitmap;

pub fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn at(k: u16, lo: u16) -> u32 {
    ((k as u32) << 16) | lo as u32
}

pub fn sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v.dedup();
    v
}

/// `keys` containers, `per_key` random lows phase-shifted — dense (bitmap) when
/// `per_key` is large, array otherwise.
pub fn dense(keys: u16, per_key: u32, phase: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in 0..keys {
        for _ in 0..per_key {
            v.push(at(k, ((splitmix64(st) as u32).wrapping_add(phase) % 65536) as u16));
        }
    }
    sorted(v)
}

/// Sparse arrays: `keys` containers of `per_key` random lows.
pub fn arrays(keys: u16, per_key: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in 0..keys {
        for _ in 0..per_key {
            v.push(at(k, (splitmix64(st) % 65536) as u16));
        }
    }
    sorted(v)
}

/// Run-heavy: consecutive blocks per container.
pub fn runs(keys: u16, block: u32, gap: u32) -> Vec<u32> {
    let mut v = Vec::new();
    for k in 0..keys {
        let mut lo = 0u32;
        while lo + block <= 65536 {
            for j in 0..block {
                v.push(at(k, (lo + j) as u16));
            }
            lo += block + gap;
        }
    }
    sorted(v)
}

/// `keys` containers of `nranges` long ranges (`rlen` each, phase-shifted) —
/// dense and run-encoded (the kernels take the native run path).
pub fn run_ranges(keys: u16, nranges: u32, rlen: u32, phase: u32) -> Vec<u32> {
    let stride = 65536 / nranges;
    let mut v = Vec::new();
    for k in 0..keys {
        for i in 0..nranges {
            let start = (i * stride + phase) % 65536;
            let end = (start + rlen).min(65535);
            for lo in start..=end {
                v.push(at(k, lo as u16));
            }
        }
    }
    sorted(v)
}

/// One workload in both representations.
pub struct Set {
    pub fbs: Vec<FrozenBitmap>,
    pub rbs: Vec<RoaringBitmap>,
}

impl Set {
    pub fn new(inputs: &[Vec<u32>]) -> Self {
        let fbs = inputs
            .iter()
            .map(|v| {
                let mut b = frostbit::FrozenBitmapBuilder::new();
                b.extend_sorted(v.iter().copied());
                b.finish()
            })
            .collect();
        // Best-vs-best: frostbit's builder auto-picks run encoding, so let
        // roaring run-optimize too (a no-op for non-run-friendly inputs).
        let rbs = inputs
            .iter()
            .map(|v| {
                let mut r: RoaringBitmap = v.iter().copied().collect();
                r.optimize();
                r
            })
            .collect();
        Set { fbs, rbs }
    }
    pub fn views(&self, n: usize) -> Vec<FrozenBitmapView<'_>> {
        self.fbs[..n].iter().map(|b| b.view()).collect()
    }
    pub fn fv(&self, i: usize) -> FrozenBitmapView<'_> {
        self.fbs[i].view()
    }
    pub fn len(&self) -> usize {
        self.fbs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fbs.is_empty()
    }
}

// Roaring N-way folds (first-minus-rest for diff), matching the frostbit ops.

pub fn rb_and(r: &[RoaringBitmap]) -> RoaringBitmap {
    let mut a = r[0].clone();
    for b in &r[1..] {
        a = &a & b;
    }
    a
}

pub fn rb_or(r: &[RoaringBitmap]) -> RoaringBitmap {
    let mut a = RoaringBitmap::new();
    for b in r {
        a = &a | b;
    }
    a
}

pub fn rb_diff(r: &[RoaringBitmap]) -> RoaringBitmap {
    let mut a = r[0].clone();
    for b in &r[1..] {
        a = &a - b;
    }
    a
}

pub fn fb_vec(b: &FrozenBitmap) -> Vec<u32> {
    b.view().iter().collect()
}
pub fn rb_vec(r: &RoaringBitmap) -> Vec<u32> {
    r.iter().collect()
}
