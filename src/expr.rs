//! Boolean expression trees over frozen bitmaps.
//!
//! [`BitmapExpr`] is a recursive *definition* — leaves (zero-copy views or
//! shared owned bitmaps) combined with AND / OR / DIFF. Because Rust builds
//! children before parents, **construction is analysis**: each combinator folds
//! its children's [`FoldPlan`]s into one flat, post-order step list, flattening
//! same-op chains by moving step references (`And(And(a, b), c)` ⇒ one
//! `intersect([a, b, c])`) and computing the exact operand-stack depth as it
//! goes. No work-stack, no recursive traversal — every node is handled exactly
//! once, and the working-set size is known up front.
//!
//! [`BitmapExpr::materialize`] runs the finished plan over a preallocated
//! operand stack, folding with the flat `*_fast` kernels.

use std::sync::Arc;

use crate::ops::arena::OpArena;
use crate::ops::cursor::ContainerCursor;
use crate::ops::source::{view_container_count, Inputs};
use crate::ops::kernels;
use crate::{FrozenBitmap, FrozenBitmapView};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    And,
    Or,
    Diff,
}

/// One linearized instruction: push a leaf, or pop `arity` operands and combine.
#[derive(Clone)]
enum Step<'a> {
    Leaf(FrozenBitmapView<'a>),
    Owned(Arc<FrozenBitmap>),
    Combine(Op, u32),
}

/// A flat, post-order evaluation plan with its exact peak operand-stack depth.
///
/// Built incrementally by the [`BitmapExpr`] combinators (construction-time
/// analysis) and run by [`FoldPlan::execute`]. Borrows the tree's leaves, so it
/// lives no longer than the expression it came from.
#[derive(Clone)]
pub struct FoldPlan<'a> {
    steps: Vec<Step<'a>>,
    max_depth: usize,
}

/// A boolean combination of frozen bitmaps.
#[derive(Clone)]
pub enum BitmapExpr<'a> {
    /// A zero-copy leaf.
    Leaf(FrozenBitmapView<'a>),
    /// An owned leaf, shared cheaply (e.g. a cached intermediate).
    Owned(Arc<FrozenBitmap>),
    /// A pre-analyzed AND / OR / DIFF subtree.
    Combined(FoldPlan<'a>),
}

impl<'a> BitmapExpr<'a> {
    pub fn leaf(view: FrozenBitmapView<'a>) -> Self {
        BitmapExpr::Leaf(view)
    }
    pub fn owned(bm: Arc<FrozenBitmap>) -> Self {
        BitmapExpr::Owned(bm)
    }
    pub fn and(children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        BitmapExpr::Combined(FoldPlan::combine(Op::And, children))
    }
    pub fn or(children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        BitmapExpr::Combined(FoldPlan::combine(Op::Or, children))
    }
    pub fn difference(lhs: BitmapExpr<'a>, rhs: BitmapExpr<'a>) -> Self {
        BitmapExpr::Combined(FoldPlan::diff(lhs, rhs))
    }

    /// Number of leaves in the tree.
    pub fn leaf_count(&self) -> usize {
        match self {
            BitmapExpr::Leaf(_) | BitmapExpr::Owned(_) => 1,
            BitmapExpr::Combined(p) => p.leaf_count(),
        }
    }

    /// Evaluate the tree into one [`FrozenBitmap`].
    pub fn materialize(&self) -> FrozenBitmap {
        match self {
            BitmapExpr::Leaf(v) => FrozenBitmap::from_bytes(v.as_bytes()).expect("valid leaf"),
            BitmapExpr::Owned(bm) => (**bm).clone(),
            BitmapExpr::Combined(plan) => plan.execute(),
        }
    }
}

impl<'a> From<FrozenBitmapView<'a>> for BitmapExpr<'a> {
    fn from(v: FrozenBitmapView<'a>) -> Self {
        BitmapExpr::Leaf(v)
    }
}
impl<'a> From<Arc<FrozenBitmap>> for BitmapExpr<'a> {
    fn from(b: Arc<FrozenBitmap>) -> Self {
        BitmapExpr::Owned(b)
    }
}

impl<'a> FoldPlan<'a> {
    /// Peak number of operands live at once during [`execute`](Self::execute) —
    /// the exact capacity its operand stack is allocated with.
    pub fn max_stack_depth(&self) -> usize {
        self.max_depth
    }

    /// Number of leaves folded by this plan.
    pub fn leaf_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s, Step::Leaf(_) | Step::Owned(_)))
            .count()
    }

    /// Fold `children` under `op`, flattening same-op sub-plans in place.
    fn combine(op: Op, children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        let mut steps = Vec::new();
        let mut arity = 0u32;
        let mut base = 0usize; // operands live on the stack before the next child
        let mut max_depth = 0usize;
        for child in children {
            let (net, depth) = splice(child, op, &mut steps);
            max_depth = max_depth.max(base + depth);
            base += net as usize;
            arity += net;
        }
        steps.push(Step::Combine(op, arity));
        // Pre-combine `base` (== arity) operands are live; the result leaves one.
        FoldPlan { steps, max_depth: max_depth.max(base).max(1) }
    }

    /// `lhs` minus `rhs` (never flattened — DIFF is not associative).
    fn diff(lhs: BitmapExpr<'a>, rhs: BitmapExpr<'a>) -> Self {
        let mut steps = Vec::new();
        let (_, d0) = splice(lhs, Op::Diff, &mut steps);
        let (_, d1) = splice(rhs, Op::Diff, &mut steps);
        steps.push(Step::Combine(Op::Diff, 2));
        FoldPlan { steps, max_depth: d0.max(1 + d1).max(2) }
    }

    /// Run the plan over a preallocated operand stack.
    ///
    /// Intermediate results stay as pooled [`OpArena`]s and are folded straight
    /// into the next op — no bitmap is serialized between nodes. Each `Combine`
    /// folds an in-place slice of the stack (no operand copy) and serializes
    /// only the single surviving arena at the end.
    pub fn execute(&self) -> FrozenBitmap {
        let mut stack: Vec<Acc<'_>> = Vec::with_capacity(self.max_depth);
        for step in &self.steps {
            match step {
                Step::Leaf(v) => stack.push(Acc::Leaf(*v)),
                Step::Owned(b) => stack.push(Acc::Leaf(b.view())),
                Step::Combine(op, arity) => {
                    let start = stack.len() - *arity as usize;
                    let result = match op {
                        Op::And => kernels::intersect_into(&stack[start..]),
                        Op::Or => kernels::union_into(&stack[start..]),
                        Op::Diff => kernels::diff_into(&stack[start..]),
                    };
                    stack.truncate(start);
                    stack.push(Acc::Arena(result));
                }
            }
        }
        match stack.pop().expect("non-empty plan") {
            Acc::Arena(a) => a.serialize(),
            Acc::Leaf(v) => FrozenBitmap::from_bytes(v.as_bytes()).expect("valid leaf"),
        }
    }
}

/// A working operand on the execution stack: an un-evaluated leaf (still bytes)
/// or a folded, pooled arena result chained directly into the next op.
enum Acc<'a> {
    Leaf(FrozenBitmapView<'a>),
    Arena(OpArena),
}

impl Inputs for [Acc<'_>] {
    #[inline]
    fn len(&self) -> usize {
        <[_]>::len(self)
    }
    #[inline]
    fn cursor(&self, i: usize) -> ContainerCursor<'_> {
        match &self[i] {
            Acc::Leaf(v) => ContainerCursor::new(v),
            Acc::Arena(a) => ContainerCursor::from_arena(a),
        }
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        match &self[i] {
            Acc::Leaf(v) => view_container_count(v),
            Acc::Arena(a) => a.container_count(),
        }
    }
}

/// Append `child`'s steps to `steps`, flattening when it is a same-op sub-plan.
/// Returns `(net operands contributed, peak stack depth during its steps)`.
fn splice<'a>(child: BitmapExpr<'a>, parent: Op, steps: &mut Vec<Step<'a>>) -> (u32, usize) {
    match child {
        BitmapExpr::Leaf(v) => {
            steps.push(Step::Leaf(v));
            (1, 1)
        }
        BitmapExpr::Owned(b) => {
            steps.push(Step::Owned(b));
            (1, 1)
        }
        BitmapExpr::Combined(mut plan) => {
            let root = plan.steps.last().map(|s| match s {
                Step::Combine(op, _) => *op,
                _ => unreachable!("a plan always ends in a Combine"),
            });
            let flatten = matches!(
                (parent, root),
                (Op::And, Some(Op::And)) | (Op::Or, Some(Op::Or))
            );
            if flatten {
                let Some(Step::Combine(_, k)) = plan.steps.pop() else { unreachable!() };
                steps.append(&mut plan.steps);
                (k, plan.max_depth)
            } else {
                steps.append(&mut plan.steps);
                (1, plan.max_depth)
            }
        }
    }
}
