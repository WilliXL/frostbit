//! Difference (AND NOT) kernels. One file per kernel; each owns the full
//! AVX-512 -> AVX2 -> SSE2 -> NEON -> scalar waterfall for its shape.

mod array_difference;
mod bitmap_difference;

pub use array_difference::array_diff;
pub use bitmap_difference::{andnot, andnot_count, andnot_into_count};
