//! Set operations: static analysis, fold kernels, and the input abstraction
//! that lets them drive frozen leaves and intermediate arenas alike.

pub mod arena;
pub mod cursor;
pub mod kernels;
pub mod plan;
mod simd;
pub mod source;
