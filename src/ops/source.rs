//! Fold inputs, unified over frozen leaves and working arenas.
//!
//! A tree evaluator folds a mix of zero-copy leaf views and intermediate
//! [`OpArena`](crate::ops::arena::OpArena) results. Rather than serialize each
//! intermediate back to bytes, an arena is read directly as an ordered
//! container source. The [`Inputs`] trait lets the planner and kernels drive
//! either kind through one code path, monomorphized per call site — no per-op
//! wrapping allocation.

use crate::ops::cursor::ContainerCursor;
use crate::FrozenBitmapView;

/// A list of fold inputs the planner and kernels iterate uniformly.
pub trait Inputs {
    fn len(&self) -> usize;
    /// Ascending-by-key cursor over input `i`.
    fn cursor(&self, i: usize) -> ContainerCursor<'_>;
    /// Container count of input `i` (drives AND seed selection).
    fn container_count(&self, i: usize) -> usize;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Inputs for [FrozenBitmapView<'_>] {
    #[inline]
    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    #[inline]
    fn cursor(&self, i: usize) -> ContainerCursor<'_> {
        ContainerCursor::new(&self[i])
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        view_container_count(&self[i])
    }
}

// Forwards so a fold can be driven from a fixed-size array or `Vec`, not just a
// slice (generic call sites binding `&I` don't auto-unsize `&[T; N]` / `&Vec<T>`
// to `&[T]`).
impl<T, const N: usize> Inputs for [T; N]
where
    [T]: Inputs,
{
    #[inline]
    fn len(&self) -> usize {
        N
    }
    #[inline]
    fn cursor(&self, i: usize) -> ContainerCursor<'_> {
        self.as_slice().cursor(i)
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        self.as_slice().container_count(i)
    }
}

impl<T> Inputs for Vec<T>
where
    [T]: Inputs,
{
    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    #[inline]
    fn cursor(&self, i: usize) -> ContainerCursor<'_> {
        self.as_slice().cursor(i)
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        self.as_slice().container_count(i)
    }
}

/// Container count of a leaf: O(1) for standard, a cheap walk for inline.
pub(crate) fn view_container_count(v: &FrozenBitmapView<'_>) -> usize {
    if v.is_inline() {
        let mut c = ContainerCursor::new(v);
        let mut n = 0;
        while c.peek_key().is_some() {
            n += 1;
            c.advance();
        }
        n
    } else {
        v.num_containers()
    }
}
