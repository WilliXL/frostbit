//! The surface callers touch: how a frozen bitmap is built, read, combined,
//! converted, and how its working memory is budgeted.
//!
//! Everything here is either public or re-exported from the crate root. The
//! machinery underneath — the fold kernels, their static analysis, and the SIMD
//! primitives — lives in [`ops`](crate::ops).

pub mod bitmap;
pub mod builder;
pub mod container;
#[cfg(feature = "roaring")]
pub mod convert;
pub mod expr;
pub mod pool;
pub mod view;
