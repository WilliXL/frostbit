//! Helpers shared by the integration tests.
//!
//! Every test binary that needs a deterministic RNG or a bitmap built from a
//! value list pulls them from here, so there is one definition of each rather
//! than a copy per file.

#![allow(dead_code)]

use frostbit::{FrozenBitmap, FrozenBitmapBuilder};

/// SplitMix64 — the deterministic generator every randomized test draws from.
/// Same seed, same sequence, on every platform.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build from ascending `values`, letting the builder pick the smallest
/// encoding (this is what a caller gets from `finish`).
pub fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

/// Build in standard container format, whatever the size — the op-ready form,
/// and the only one with a container index to inspect.
#[cfg(feature = "internals")]
pub fn build_standard(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish_standard()
}

/// The value in container `key` at low bits `lo`.
pub fn at(key: u16, lo: u16) -> u32 {
    ((key as u32) << 16) | lo as u32
}
