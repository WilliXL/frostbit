//! When a balanced block merge beats a scan of the larger side.

/// At or below this size ratio the arrays are balanced enough that the
/// shuffle-merge's 8-at-a-time compare wins (one horizontal reduction per block,
/// not per element). Above it, one side is "rare" — a scan of the larger wins.
pub(crate) const MERGE_MAX_RATIO: usize = 4;
