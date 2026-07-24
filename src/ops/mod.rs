//! Set operations: static analysis, fold kernels, and the input abstraction
//! that lets them drive frozen leaves and intermediate arenas alike.

pub mod arena;
pub mod cursor;
mod decide;
pub mod keymask;
pub mod kernels;
pub mod plan;
pub mod run;
pub mod shape;
pub mod source;
