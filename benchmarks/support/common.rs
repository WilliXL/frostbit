//! Shared benchmark helpers: deterministic inputs and matched frozen/roaring
//! sets, so every bench compares the engines on identical data.
//!
//! roaring's `simd` feature is nightly-only and off by default, so it is a
//! *distinct competitor*: bench IDs carry the variant ([`RB`]), and
//! `benchmarks/run.sh` runs both feature sets into one criterion directory,
//! which `benchmarks/report.py` renders as combined tables.
#![allow(dead_code)]

use frostbit::{FrozenBitmap, FrozenBitmapView};
use roaring::{MultiOps, RoaringBitmap};

/// Bench-ID label for the roaring competitor in this build: its scalar default
/// or its nightly `simd` kernels, depending on the compiled feature set.
pub const RB: &str = if cfg!(feature = "roaring-simd") { "roaring-simd" } else { "roaring" };

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

/// `per_key` random lows in each of `keys` containers starting at `k0` — the
/// key-structure knob. Two bands that share no keys make a disjoint fold; a
/// band nested inside another makes a containment fold.
pub fn key_band(k0: u16, keys: u16, per_key: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in k0..k0.saturating_add(keys) {
        for _ in 0..per_key {
            v.push(at(k, (splitmix64(st) % 65536) as u16));
        }
    }
    sorted(v)
}

/// Thinly spread: `keys` containers of `per_key` values. With `per_key` tiny
/// the builder picks the inline (FI) encoding — 4 bytes/value beats a whole
/// container header — so this is how an operand comes back inline.
pub fn thin(keys: u16, per_key: u32, phase: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in 0..keys {
        for _ in 0..per_key {
            v.push(at(k, ((splitmix64(st) as u32).wrapping_add(phase) % 65536) as u16));
        }
    }
    sorted(v)
}

/// `keys` containers of exactly `nruns` evenly spaced runs, each half the
/// stride so the gaps never close and the count holds at any `phase`. Straddle
/// `MAX_RUNS` with this to land either side of the run→bitmap decision.
pub fn run_count(keys: u16, nruns: u32, phase: u32) -> Vec<u32> {
    let stride = (65_536 / nruns).max(2);
    let rlen = stride / 2;
    let shift = phase % (stride - rlen);
    let mut v = Vec::new();
    for k in 0..keys {
        for i in 0..nruns {
            let start = i * stride + shift;
            for lo in start..(start + rlen).min(65_536) {
                v.push(at(k, lo as u16));
            }
        }
    }
    sorted(v)
}

/// `keys` fully saturated containers (all 65,536 lows) from `k0` — the one
/// shape where every kernel's output is its input.
pub fn full_keys(k0: u16, keys: u16) -> Vec<u32> {
    let mut v = Vec::new();
    for k in k0..k0.saturating_add(keys) {
        v.extend((0..65_536u32).map(|lo| at(k, lo as u16)));
    }
    v
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
        let rbs = inputs.iter().map(|v| v.iter().copied().collect()).collect();
        Set { fbs, rbs }
    }

    /// Run-optimize the roaring side (frostbit's builder already auto-encodes
    /// runs). Use only where it helps roaring — e.g. dedicated run workloads;
    /// on mixed trees roaring's run-container ops are slower than its bitmaps.
    pub fn optimize_roaring(&mut self) {
        for r in &mut self.rbs {
            r.optimize();
        }
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

// Roaring N-way ops via `MultiOps` — the library's documented fast path for
// merging many bitmaps (first-minus-rest for diff), matching the frostbit ops.

pub fn rb_and(r: &[RoaringBitmap]) -> RoaringBitmap {
    r.iter().intersection()
}

pub fn rb_or(r: &[RoaringBitmap]) -> RoaringBitmap {
    r.iter().union()
}

pub fn rb_diff(r: &[RoaringBitmap]) -> RoaringBitmap {
    r.iter().difference()
}

pub fn fb_vec(b: &FrozenBitmap) -> Vec<u32> {
    b.view().iter().collect()
}
pub fn rb_vec(r: &RoaringBitmap) -> Vec<u32> {
    r.iter().collect()
}
