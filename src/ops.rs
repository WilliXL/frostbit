//! Set operations: static analysis, fold kernels, and the input abstraction
//! that lets them drive frozen leaves and intermediate arenas alike.

pub mod arena;
pub mod analyze;
pub mod cursor;
pub mod keymask;
pub mod kernels;
// SIMD kernels are public under `internals` for white-box kernel benchmarks.
#[cfg(feature = "internals")]
pub mod simd;
#[cfg(not(feature = "internals"))]
mod simd;
pub mod source;
