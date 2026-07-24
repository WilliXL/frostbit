//! Set operations: static analysis, fold kernels, and the input abstraction
//! that lets them drive frozen leaves and intermediate arenas alike.

pub mod arena;
pub mod analyze;
pub mod cursor;
pub mod keymask;
pub mod kernels;
pub mod run;
pub mod source;
