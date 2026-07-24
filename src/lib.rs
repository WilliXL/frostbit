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
pub use expr::{BitmapExpr, FoldPlan};

/// Set ops + their static analysis pass. The module is public only under
/// `internals`; the stable entry points are re-exported below.
#[cfg(feature = "internals")]
pub mod ops;
#[cfg(not(feature = "internals"))]
mod ops;

/// Op-ready (`_fast`) set operations: results are standard-format, ready to
/// feed the next op. Use the (forthcoming) `_compact` variants for storage.
pub use ops::kernels::{diff as difference_fast, intersect as intersect_fast, union as union_fast};

#[cfg(feature = "roaring")]
mod convert;
