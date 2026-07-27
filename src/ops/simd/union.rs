//! Union (OR) kernels. One file per kernel; each owns the full
//! AVX-512 -> AVX2 -> SSE2 -> NEON -> scalar waterfall for its shape.

mod array_union;
mod bitmap_union;

pub use array_union::array_union;
pub use bitmap_union::{or, or_count};
