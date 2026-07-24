//! Frostbit — frozen, mmap-friendly, zero-copy roaring bitmaps.
//!
//! A frozen bitmap is a roaring bitmap serialized into a compact, aligned byte
//! buffer that's queried directly from raw bytes (e.g. an `mmap`) with no
//! deserialization — the read-optimized counterpart to [`roaring::RoaringBitmap`].
//!
//! Op producers come in two intents: `_fast` (op-ready, for query pipelines)
//! and `_compact` (smallest, for persistence). The builder always finishes
//! compact — built bitmaps are destined for storage.

/// Wire-format constants and byte primitives. Public only under `internals`.
#[cfg(feature = "internals")]
pub mod format;
#[cfg(not(feature = "internals"))]
mod format;

mod bitmap;
pub use bitmap::FrozenBitmap;

/// Container payload access. Public only under `internals`.
#[cfg(feature = "internals")]
pub mod container;
#[cfg(not(feature = "internals"))]
mod container;

mod builder;
pub use builder::FrozenBitmapBuilder;

mod view;
pub use view::{FrozenBitmapView, Iter};

mod expr;
pub use expr::BitmapExpr;

/// Set ops + their static analysis pass. The module is public only under
/// `internals`; the stable entry points are re-exported below.
#[cfg(feature = "internals")]
pub mod ops;
#[cfg(not(feature = "internals"))]
mod ops;

/// Free set operations in two intents: `_fast` results are op-ready standard
/// container form (ideal for feeding the next op); `_compact` results are the
/// smallest form (ideal for persistence). Both fold identically — they differ
/// only in how the result is serialized.
pub use ops::kernels::{
    diff as difference_fast, diff_compact as difference_compact, intersect as intersect_fast,
    intersect_compact, union as union_fast, union_compact,
};

#[cfg(feature = "roaring")]
mod convert;
