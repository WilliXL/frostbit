//! Fold kernels, one file per set operation.
//!
//! Each writes only into pre-sized arena slots (the arena's `record`
//! debug-asserts the no-runtime-allocation invariant) and dispatches on the
//! typed [`Data`](crate::container::Data) view, delegating the heavy lifting to
//! [`super::simd`].

mod accum;
mod difference;
pub mod run;
mod intersect;
mod union;

// Exported under the public-facing names: a bare `intersect`/`union` would
// collide with the module of the same name in a `use` path.
pub use difference::{diff as difference_fast, diff_compact as difference_compact};
pub use difference::diff_fold;
#[cfg(feature = "internals")]
pub use difference::diff_into;
pub use intersect::{intersect as intersect_fast, intersect_compact};
pub use intersect::intersect_fold;
#[cfg(feature = "internals")]
pub use intersect::intersect_into;
pub use union::{union as union_fast, union_compact};
pub use union::union_fold;
#[cfg(feature = "internals")]
pub use union::union_into;
