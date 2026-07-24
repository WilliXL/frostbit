//! Primitives with more than one caller. Anything used by a single kernel
//! lives with that kernel instead.

mod block;
mod compact;
mod fold;
mod popcount;
mod window_scan;
mod words;

pub(crate) use block::MERGE_MAX_RATIO;
pub(crate) use compact::COMPACT;
#[cfg(target_arch = "aarch64")]
pub(crate) use fold::{fold_count_neon, fold_neon};
pub use popcount::popcount;
#[cfg(target_arch = "aarch64")]
pub(crate) use window_scan::window_has_neon;
#[cfg(target_arch = "x86_64")]
pub(crate) use window_scan::{window_has_avx2, window_has_sse2};
pub use words::{clear, clear_runs, clear_values, copy, set_runs, set_values};
