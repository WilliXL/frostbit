//! Boolean expression trees over frozen bitmaps.
//!
//! [`BitmapExpr`] is a recursive *definition* — leaves (zero-copy views or
//! shared owned bitmaps) combined with AND / OR / DIFF. Because Rust builds
//! children before parents, **construction is analysis**: each combinator folds
//! its children into one flat, post-order step list, flattening same-op chains
//! (`And(And(a, b), c)` ⇒ one `intersect([a, b, c])`) and, crucially,
//! propagating each node's output *shape* bottom-up so every op's arena plan
//! (keys + slot byte-ceilings) is computed **once, up front**.
//!
//! [`BitmapExpr::materialize`] then runs that manifest: the executor sizes each
//! arena straight from the precomputed plan and folds — it does no sizing
//! analysis of its own.

use std::sync::Arc;

use crate::ops::arena::OpArena;
use crate::ops::cursor::ContainerCursor;
use crate::ops::keymask::KeyMask;
use crate::ops::kernels;
use crate::ops::plan::{Op as PlanOp, Plan};
use crate::ops::shape::{self, view_shape, Shape};
use crate::ops::source::{view_container_count, Inputs};
use crate::{FrozenBitmap, FrozenBitmapView};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    And,
    Or,
    Diff, // only used as a never-flatten parent for DIFF operands
}

/// One linearized instruction: push a leaf, or pop `arity` operands and fold
/// them with a fully precomputed arena [`Plan`].
#[derive(Clone)]
enum Step<'a> {
    Leaf(FrozenBitmapView<'a>),
    Owned(Arc<FrozenBitmap>),
    Combine(u32, Plan),
}

/// A flat, post-order evaluation manifest: the step list, this subtree's output
/// [`Shape`] (for the parent's analysis), and the peak operand-stack depth.
///
/// Built once by the [`BitmapExpr`] combinators; run by [`FoldPlan::execute`]
/// with no further analysis. Borrows the tree's leaves.
#[derive(Clone)]
pub struct FoldPlan<'a> {
    steps: Vec<Step<'a>>,
    shape: Shape,
    max_depth: usize,
    /// Hole-punch mask (set by [`BitmapExpr::punch_holes`]): the root's surviving
    /// container keys, applied to every leaf cursor so dead blocks are skipped.
    live: Option<Arc<KeyMask>>,
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

    /// Enable hole-punching. When this is a root intersection (AND of ≥2
    /// operands), derive the surviving container-key set and prune every leaf
    /// cursor to it — dead 64K blocks are skipped before any fold, so an
    /// `AND(narrow, OR(wide…))` never materializes the wide branch's dead
    /// blocks. A no-op for any other shape, where no key set narrows.
    pub fn punch_holes(self) -> Self {
        match self {
            BitmapExpr::Combined(mut fp) if fp.is_and_root() => {
                let mut mask = KeyMask::empty();
                for m in &fp.shape {
                    mask.set(m.key);
                }
                fp.live = Some(Arc::new(mask));
                BitmapExpr::Combined(fp)
            }
            other => other,
        }
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
    /// Peak number of operands live at once during [`execute`](Self::execute).
    pub fn max_stack_depth(&self) -> usize {
        self.max_depth
    }

    /// True when the root fold is an intersection of ≥2 operands — the only
    /// shape whose key set narrows what its branches can contribute (a union or
    /// lone leaf spans all its leaves' keys, so a mask would prune nothing).
    fn is_and_root(&self) -> bool {
        matches!(
            self.steps.last(),
            Some(Step::Combine(arity, plan)) if *arity >= 2 && plan.op == PlanOp::Intersect
        )
    }

    /// Number of leaves folded by this plan.
    pub fn leaf_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| matches!(s, Step::Leaf(_) | Step::Owned(_)))
            .count()
    }

    /// Fold `children` under `op`, flattening same-op sub-plans and computing the
    /// output shape (and thus this op's arena plan) from the children's shapes.
    fn combine(op: Op, children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        let mut steps = Vec::new();
        let mut shapes: Vec<Shape> = Vec::new();
        let (mut arity, mut base, mut max_depth) = (0u32, 0usize, 0usize);
        for child in children {
            let (shape, net, depth) = splice(child, op, &mut steps);
            shapes.push(shape);
            max_depth = max_depth.max(base + depth);
            base += net as usize;
            arity += net;
        }
        let (pop, shape) = match op {
            Op::And => (PlanOp::Intersect, shape::intersect_shape(&shapes)),
            Op::Or => (PlanOp::Union, shape::union_shape(&shapes)),
            Op::Diff => unreachable!("combine is AND/OR only"),
        };
        steps.push(Step::Combine(arity, shape::to_plan(pop, &shape)));
        FoldPlan { steps, shape, max_depth: max_depth.max(base).max(1), live: None }
    }

    /// `lhs` minus `rhs` (never flattened — DIFF is not associative).
    fn diff(lhs: BitmapExpr<'a>, rhs: BitmapExpr<'a>) -> Self {
        let mut steps = Vec::new();
        let (s0, _, d0) = splice(lhs, Op::Diff, &mut steps); // Op::Diff never flattens
        let (s1, _, d1) = splice(rhs, Op::Diff, &mut steps);
        let shape = shape::diff_shape(&[s0, s1]);
        steps.push(Step::Combine(2, shape::to_plan(PlanOp::Diff, &shape)));
        FoldPlan { steps, shape, max_depth: d0.max(1 + d1).max(2), live: None }
    }

    /// Run the manifest over a preallocated operand stack. Each `Combine` sizes
    /// its arena from the precomputed plan and folds an in-place stack slice (no
    /// operand copy, no sizing analysis); only the final arena is serialized.
    /// The operand stack itself is pooled, so a materialize allocates only its
    /// result.
    pub fn execute(&self) -> FrozenBitmap {
        let mut guard = ExecStack::take();
        let stack = guard.borrow();
        stack.reserve(self.max_depth);
        let mask = self.live.as_deref();
        for step in &self.steps {
            match step {
                Step::Leaf(v) => stack.push(Acc::Leaf(*v)),
                Step::Owned(b) => stack.push(Acc::Leaf(b.view())),
                Step::Combine(arity, plan) => {
                    let start = stack.len() - *arity as usize;
                    let mut arena = OpArena::from_plan(plan);
                    let inputs = Masked { accs: &stack[start..], mask };
                    match plan.op {
                        PlanOp::Intersect => kernels::intersect_fold(&mut arena, &inputs),
                        PlanOp::Union => kernels::union_fold(&mut arena, &inputs),
                        PlanOp::Diff => kernels::diff_fold(&mut arena, &inputs),
                    }
                    stack.truncate(start);
                    stack.push(Acc::Arena(arena));
                }
            }
        }
        // Terminal result → compact (smallest), like roaring's output.
        match stack.pop().expect("non-empty plan") {
            Acc::Arena(a) => a.serialize_compact(),
            Acc::Leaf(v) => FrozenBitmap::from_bytes(v.as_bytes()).expect("valid leaf"),
        }
    }
}

/// Pooled operand-stack buffer for [`FoldPlan::execute`], reused across
/// materializations. Lifetime-erased in the pool (only ever stored empty),
/// loaned at the leaves' lifetime — exactly like the arena and fold pools.
struct ExecStack {
    v: Vec<Acc<'static>>,
}

mod stack_pool {
    use std::cell::RefCell;

    use super::Acc;

    const MAX_POOLED: usize = 8;
    thread_local! {
        static POOL: RefCell<Vec<Vec<Acc<'static>>>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn take() -> Vec<Acc<'static>> {
        POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
    }

    pub(super) fn put(v: Vec<Acc<'static>>) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < MAX_POOLED {
                p.push(v);
            }
        });
    }
}

impl ExecStack {
    #[inline]
    fn take() -> Self {
        ExecStack { v: stack_pool::take() }
    }

    /// Borrow the (empty) stack buffer relabeled to the leaves' lifetime `'a`.
    #[inline]
    fn borrow<'a>(&mut self) -> &mut Vec<Acc<'a>> {
        debug_assert!(self.v.is_empty());
        // SAFETY: the buffer is empty across every pool boundary, so no `'static`
        // Acc is materialized — we only relabel the empty buffer to the caller's
        // `'a`. Operands pushed during execute are drained before it returns (or
        // dropped by `clear` on unwind), so none outlives what it borrows.
        unsafe { &mut *(&mut self.v as *mut Vec<Acc<'static>> as *mut Vec<Acc<'a>>) }
    }
}

impl Drop for ExecStack {
    fn drop(&mut self) {
        self.v.clear();
        stack_pool::put(std::mem::take(&mut self.v));
    }
}

/// A working operand on the execution stack: an un-evaluated leaf (still bytes)
/// or a folded, pooled arena result chained directly into the next op.
enum Acc<'a> {
    Leaf(FrozenBitmapView<'a>),
    Arena(OpArena),
}

/// One `Combine`'s operand slice, optionally hole-punched. Leaf cursors skip
/// keys absent from `mask`; intermediate arenas are already pruned (their leaves
/// were masked when they were folded), so they read back verbatim.
struct Masked<'a, 'm> {
    accs: &'a [Acc<'a>],
    mask: Option<&'m KeyMask>,
}

impl Inputs for Masked<'_, '_> {
    #[inline]
    fn len(&self) -> usize {
        self.accs.len()
    }
    #[inline]
    fn cursor(&self, i: usize) -> ContainerCursor<'_> {
        match (&self.accs[i], self.mask) {
            (Acc::Leaf(v), Some(mask)) => ContainerCursor::new_live(v, mask),
            (Acc::Leaf(v), None) => ContainerCursor::new(v),
            (Acc::Arena(a), _) => ContainerCursor::from_arena(a),
        }
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        match &self.accs[i] {
            Acc::Leaf(v) => view_container_count(v),
            Acc::Arena(a) => a.container_count(),
        }
    }
}

/// Append `child`'s steps to `steps`, flattening when it is a same-op sub-plan,
/// and yield its output [`Shape`] for the parent's analysis (moved out of a
/// sub-plan, never cloned). Returns `(shape, net operands contributed, peak
/// stack depth during its steps)`.
fn splice<'a>(child: BitmapExpr<'a>, parent: Op, steps: &mut Vec<Step<'a>>) -> (Shape, u32, usize) {
    match child {
        BitmapExpr::Leaf(v) => {
            let shape = view_shape(&v);
            steps.push(Step::Leaf(v));
            (shape, 1, 1)
        }
        BitmapExpr::Owned(b) => {
            let shape = view_shape(&b.view());
            steps.push(Step::Owned(b));
            (shape, 1, 1)
        }
        BitmapExpr::Combined(mut fp) => {
            let shape = std::mem::take(&mut fp.shape);
            let root = fp.steps.last().map(|s| match s {
                Step::Combine(_, p) => p.op,
                _ => unreachable!("a plan always ends in a Combine"),
            });
            let flatten = matches!(
                (parent, root),
                (Op::And, Some(PlanOp::Intersect)) | (Op::Or, Some(PlanOp::Union))
            );
            if flatten {
                let Some(Step::Combine(k, _)) = fp.steps.pop() else { unreachable!() };
                steps.append(&mut fp.steps);
                (shape, k, fp.max_depth)
            } else {
                steps.append(&mut fp.steps);
                (shape, 1, fp.max_depth)
            }
        }
    }
}
