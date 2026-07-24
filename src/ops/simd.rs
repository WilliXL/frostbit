//! SIMD container kernels, one folder per set operation and one file per
//! kernel. Each kernel file owns its whole dispatch waterfall:
//!
//! ```text
//! AVX-512  ->  AVX2  ->  SSE2  ->  NEON  ->  scalar
//! ```
//!
//! `cfg(target_arch)` selects at compile time; on x86 `is_x86_feature_detected!`
//! selects among the ISAs at runtime (a cached load). NEON is baseline on
//! aarch64, so that path has no runtime check. There is always a portable
//! scalar fallback, so the same code is correct on every target.

pub mod common;
mod difference;
mod intersect;
mod union;

pub use common::{clear, clear_runs, clear_values, copy, popcount, set_runs, set_values};
pub use difference::{andnot, andnot_count, andnot_into_count, array_diff};
pub use intersect::{and_count, array_intersect};
pub use union::{array_union, or, or_count};
