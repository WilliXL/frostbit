//! SIMD container kernels, grouped by set operation.
//!
//! Each op exposes one safe entry point that picks the best available
//! implementation, compile-gated by `cfg(target_arch)` and (on x86) selected at
//! runtime by `is_x86_feature_detected!`, in order:
//!
//! ```text
//! AVX-512  →  AVX2  →  SSE2  →  NEON  →  scalar
//! ```
//!
//! SSE2 is baseline on `x86_64` and NEON is baseline on `aarch64`, so those
//! rungs need no runtime check. There is always a portable scalar fallback, so
//! the same code is correct on every target.

use crate::container::Bitmap;

mod common;
mod difference;
mod intersect;
mod union;

pub use common::{clear, clear_runs, clear_values, copy, popcount, set_runs, set_values};
pub use difference::{andnot, andnot_count, andnot_into_count, array_diff};
pub use intersect::{and_count, array_intersect};
pub use union::{array_union, or, or_count};
