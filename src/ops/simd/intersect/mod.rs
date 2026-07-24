//! Intersection (AND) kernels. One file per kernel; each owns the full
//! AVX-512 -> AVX2 -> SSE2 -> NEON -> scalar waterfall for its shape.

mod array_intersect;
mod bitmap_intersect;

pub use array_intersect::array_intersect;
pub use bitmap_intersect::and_count;
