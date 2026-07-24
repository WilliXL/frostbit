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

use std::sync::{Arc, OnceLock};

use crate::ops::arena::OpArena;
use crate::ops::cursor::ContainerCursor;
use crate::ops::keymask::KeyMask;
use crate::ops::kernels;
use crate::ops::analyze::plan::{Op as PlanOp, Plan};
use crate::ops::analyze::shape::{self, view_shape, Shape};
use crate::ops::source::{view_container_count, Inputs};
use crate::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};

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
    /// Short-circuit guard for an AND/DIFF: if the operand just produced (top of
    /// stack) is empty, the whole op is empty — drop its `pop` partial operands,
    /// push an empty result, and jump `skip` steps forward (past the fold),
    /// skipping the remaining operands' evaluation. `skip` is a *relative* offset
    /// so it survives this subtree being spliced (index-shifted) into a parent.
    Guard { skip: u32, pop: u32 },
}

/// A flat, post-order evaluation manifest: the step list, this subtree's output
/// [`Shape`] (for the parent's analysis), and the peak operand-stack depth.
///
/// Built once by the [`BitmapExpr`] combinators; run by [`FoldPlan::execute`]
/// with no further analysis. Borrows the tree's leaves.
#[derive(Clone)]
struct FoldPlan<'a> {
    steps: Vec<Step<'a>>,
    shape: Shape,
    max_depth: usize,
    /// Whether this node is an AND whose key set is narrower than some child's
    /// — the only shape where a hole-punch mask prunes anything.
    narrows: bool,
    /// The mask itself, built on first use. Only the *root* ever runs, and
    /// splicing into a parent discards a child's mask, so deriving it during
    /// analysis would allocate and fill 8 KiB for every interior AND and throw
    /// all but one away. Built once here, then reused by every materialize.
    live: OnceLock<Option<Arc<KeyMask>>>,
}

/// A boolean combination of frozen bitmaps. Build it with the
/// [`leaf`](Self::leaf) / [`owned`](Self::owned) / [`and`](Self::and) /
/// [`or`](Self::or) / [`difference`](Self::difference) constructors, then
/// evaluate with [`materialize`](Self::materialize). The internal form is an
/// opaque implementation detail.
#[derive(Clone)]
pub struct BitmapExpr<'a>(Node<'a>);

#[derive(Clone)]
enum Node<'a> {
    /// A zero-copy leaf.
    Leaf(FrozenBitmapView<'a>),
    /// An owned leaf, shared cheaply (e.g. a cached intermediate).
    Owned(Arc<FrozenBitmap>),
    /// A pre-analyzed AND / OR / DIFF subtree.
    Combined(FoldPlan<'a>),
}

impl<'a> BitmapExpr<'a> {
    /// A zero-copy leaf borrowing a view.
    pub fn leaf(view: FrozenBitmapView<'a>) -> Self {
        BitmapExpr(Node::Leaf(view))
    }

    /// A shared owned leaf. For a single-use owned bitmap prefer
    /// `BitmapExpr::from(bm)`; this `Arc` form is for cheap sharing across trees.
    pub fn owned(bm: Arc<FrozenBitmap>) -> Self {
        BitmapExpr(Node::Owned(bm))
    }

    /// Intersection (AND) of the children, flattening nested ANDs into one N-way
    /// op. Degenerate inputs: `and([])` is the empty set and `and([x])` is `x`.
    pub fn and(children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        BitmapExpr(Node::Combined(FoldPlan::combine(Op::And, children)))
    }

    /// Union (OR) of the children, flattening nested ORs into one N-way op.
    /// Degenerate inputs: `or([])` is the empty set and `or([x])` is `x`.
    pub fn or(children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        BitmapExpr(Node::Combined(FoldPlan::combine(Op::Or, children)))
    }

    /// Difference: `lhs` minus `rhs`. Never flattened — DIFF is not associative.
    pub fn difference(lhs: BitmapExpr<'a>, rhs: BitmapExpr<'a>) -> Self {
        BitmapExpr(Node::Combined(FoldPlan::diff(lhs, rhs)))
    }

    /// Evaluate the tree into one [`FrozenBitmap`].
    pub fn materialize(&self) -> FrozenBitmap {
        match &self.0 {
            Node::Leaf(v) => FrozenBitmap::from_bytes_trusted(v.as_bytes()),
            Node::Owned(bm) => (**bm).clone(),
            Node::Combined(plan) => plan.execute(),
        }
    }
}

impl<'a> From<FrozenBitmapView<'a>> for BitmapExpr<'a> {
    fn from(v: FrozenBitmapView<'a>) -> Self {
        BitmapExpr(Node::Leaf(v))
    }
}
impl<'a> From<Arc<FrozenBitmap>> for BitmapExpr<'a> {
    fn from(b: Arc<FrozenBitmap>) -> Self {
        BitmapExpr(Node::Owned(b))
    }
}
impl<'a> From<FrozenBitmap> for BitmapExpr<'a> {
    /// Place an owned bitmap in a tree by value (wrapped in an `Arc` internally),
    /// so single-use owned leaves don't need an explicit `Arc::new`.
    fn from(bm: FrozenBitmap) -> Self {
        BitmapExpr(Node::Owned(Arc::new(bm)))
    }
}

impl<'a> FoldPlan<'a> {
    /// Fold `children` under `op`, flattening same-op sub-plans and computing the
    /// output shape (and thus this op's arena plan) from the children's shapes.
    fn combine(op: Op, children: impl IntoIterator<Item = BitmapExpr<'a>>) -> Self {
        let mut steps = Vec::new();
        let mut shapes: Vec<Shape> = Vec::new();
        let mut weights: Vec<usize> = Vec::new();
        let (mut arity, mut base, mut max_depth) = (0u32, 0usize, 0usize);
        for child in children {
            let (shape, net, depth, guardable) = splice(child, op, &mut steps);
            shapes.push(shape);
            weights.push(net as usize);
            max_depth = max_depth.max(base + depth);
            base += net as usize;
            arity += net;
            // After each subtree operand of an AND, guard: if it evaluated empty
            // the whole AND is empty, so skip the rest. `skip_to` is patched to
            // jump past the fold once its index is known.
            if op == Op::And && guardable {
                steps.push(Step::Guard { skip: u32::MAX, pop: base as u32 });
            }
        }
        let (pop, shape) = match op {
            Op::And => (PlanOp::Intersect, shape::intersect_shape(&shapes)),
            Op::Or => (PlanOp::Union, shape::union_shape(&shapes, &weights)),
            Op::Diff => unreachable!("combine is AND/OR only"),
        };
        // Auto hole-punch applies when this AND's intersected key set is
        // narrower than some child's: a mask over the surviving keys provably
        // skips dead blocks in the wider branches. Record *that* here (it is a
        // length comparison); the mask is built lazily, since only the root's
        // is ever used.
        let narrows = op == Op::And
            && arity >= 2
            && shapes.iter().map(Vec::len).max().is_some_and(|w| shape.len() < w);
        let after = steps.len() + 1;
        steps.push(Step::Combine(arity, shape::to_plan(pop, &shape)));
        patch_guards(&mut steps, after);
        FoldPlan {
            steps,
            shape,
            max_depth: max_depth.max(base).max(1),
            narrows,
            live: OnceLock::new(),
        }
    }

    /// `lhs` minus `rhs` (never flattened — DIFF is not associative).
    fn diff(lhs: BitmapExpr<'a>, rhs: BitmapExpr<'a>) -> Self {
        let mut steps = Vec::new();
        let (s0, _, d0, lhs_guardable) = splice(lhs, Op::Diff, &mut steps); // Op::Diff never flattens
        // An empty lhs makes the whole difference empty (the rhs only removes),
        // so guard it and skip evaluating the rhs.
        if lhs_guardable {
            steps.push(Step::Guard { skip: u32::MAX, pop: 1 });
        }
        let (s1, _, d1, _) = splice(rhs, Op::Diff, &mut steps);
        let shape = shape::diff_shape(&[s0, s1]);
        let after = steps.len() + 1;
        steps.push(Step::Combine(2, shape::to_plan(PlanOp::Diff, &shape)));
        patch_guards(&mut steps, after);
        FoldPlan {
            steps,
            shape,
            max_depth: d0.max(1 + d1).max(2),
            narrows: false,
            live: OnceLock::new(),
        }
    }

    /// Run the manifest over a preallocated operand stack. Each `Combine` sizes
    /// its arena from the precomputed plan and folds an in-place stack slice (no
    /// operand copy, no sizing analysis); only the final arena is serialized.
    /// The operand stack itself is pooled, so a materialize allocates only its
    /// result.
    fn execute(&self) -> FrozenBitmap {
        let mut guard = ExecStack::take();
        let stack = guard.borrow();
        stack.reserve(self.max_depth);
        // Derive the hole-punch mask on first run and keep it for every later
        // materialize (see `live`).
        let mask = self
            .live
            .get_or_init(|| {
                self.narrows.then(|| {
                    let mut m = KeyMask::empty();
                    for meta in &self.shape {
                        m.set(meta.key);
                    }
                    Arc::new(m)
                })
            })
            .as_deref();
        let mut pc = 0;
        while pc < self.steps.len() {
            match &self.steps[pc] {
                Step::Leaf(v) => stack.push(Acc::Leaf(*v)),
                Step::Owned(b) => stack.push(Acc::Leaf(b.view())),
                Step::Guard { skip, pop } => {
                    if stack.last().is_some_and(Acc::is_empty) {
                        let keep = stack.len() - *pop as usize;
                        stack.truncate(keep);
                        stack.push(Acc::Empty);
                        pc += *skip as usize;
                        continue;
                    }
                }
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
            pc += 1;
        }
        // Terminal result → compact (smallest), like roaring's output.
        match stack.pop().expect("non-empty plan") {
            Acc::Arena(a) => a.serialize_compact(),
            Acc::Leaf(v) => FrozenBitmap::from_bytes_trusted(v.as_bytes()),
            Acc::Empty => FrozenBitmapBuilder::new().finish(),
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
    use super::Acc;
    use crate::pool::Pool;

    thread_local! {
        static POOL: Pool<Vec<Acc<'static>>> = const { Pool::new("operand-stack") };
    }

    pub(super) fn take() -> Vec<Acc<'static>> {
        POOL.with(|p| p.take(Vec::new))
    }

    pub(super) fn put(v: Vec<Acc<'static>>) {
        POOL.with(|p| p.put(v));
    }

    pub(crate) fn clear() {
        POOL.with(Pool::clear);
    }
}

pub(crate) use stack_pool::clear as clear_stack_pool;

impl ExecStack {
    #[inline]
    fn take() -> Self {
        ExecStack { v: stack_pool::take() }
    }

    /// Borrow the (empty) stack buffer relabeled to the leaves' lifetime `'a`.
    // The second pointer cast only changes the element lifetime, which clippy
    // can't see (lifetimes are erased in the cast type), so it reads as a
    // redundant same-type cast — but it is load-bearing: dropping it (clippy's
    // suggestion) yields a `Vec<Acc<'static>>` and fails to compile.
    #[allow(clippy::unnecessary_cast)]
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
    /// A short-circuited (empty) operand or result.
    Empty,
}

impl Acc<'_> {
    #[inline]
    fn is_empty(&self) -> bool {
        match self {
            Acc::Leaf(v) => view_container_count(v) == 0,
            Acc::Arena(a) => a.is_empty(),
            Acc::Empty => true,
        }
    }
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
            (Acc::Empty, _) => ContainerCursor::empty(),
        }
    }
    #[inline]
    fn container_count(&self, i: usize) -> usize {
        match &self.accs[i] {
            Acc::Leaf(v) => view_container_count(v),
            Acc::Arena(a) => a.container_count(),
            Acc::Empty => 0,
        }
    }
}

/// Append `child`'s steps to `steps`, flattening when it is a same-op sub-plan,
/// and yield its output [`Shape`] for the parent's analysis (moved out of a
/// sub-plan, never cloned). Returns `(shape, net operands contributed, peak
/// stack depth during its steps)`.
fn splice<'a>(child: BitmapExpr<'a>, parent: Op, steps: &mut Vec<Step<'a>>) -> (Shape, u32, usize, bool) {
    match child.0 {
        Node::Leaf(v) => {
            let shape = view_shape(&v);
            steps.push(Step::Leaf(v));
            (shape, 1, 1, false)
        }
        Node::Owned(b) => {
            let shape = view_shape(&b.view());
            steps.push(Step::Owned(b));
            (shape, 1, 1, false)
        }
        Node::Combined(mut fp) => {
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
                // Inlining moves these steps up, so the child's own guards (which
                // jumped to its now-popped Combine) no longer apply — drop them.
                // The flattened operands are guarded, if at all, by this op.
                fp.steps.retain(|s| !matches!(s, Step::Guard { .. }));
                steps.append(&mut fp.steps);
                (shape, k, fp.max_depth, false)
            } else {
                // A non-flattened subtree: a single operand whose result may be
                // empty — the caller guards it.
                steps.append(&mut fp.steps);
                (shape, 1, fp.max_depth, true)
            }
        }
    }
}

/// Resolve every not-yet-targeted guard (`skip == u32::MAX`) in `steps` to a
/// relative offset landing on `target` (the step past the enclosing fold), once
/// that index is known. Guards belonging to nested, non-flattened subtrees are
/// already resolved (to their own folds) — leave them.
fn patch_guards(steps: &mut [Step<'_>], target: usize) {
    for (i, s) in steps.iter_mut().enumerate() {
        if let Step::Guard { skip, .. } = s {
            if *skip == u32::MAX {
                *skip = (target - i) as u32;
            }
        }
    }
}
