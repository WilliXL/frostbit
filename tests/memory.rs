//! A warm fold must not allocate.
//!
//! Every working buffer (op arenas, fold scratch, the operand stack, result
//! buffers) comes from a per-thread pool, and an expression tree is evaluated by
//! walking one flat, contiguous step list with a program counter — no recursion,
//! no per-node worklist. So once the pools are warm, a repeated `materialize` /
//! `*_fast` should perform *zero* heap allocations. This test pins that down
//! with a counting global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use frostbit::{intersect_fast, union_fast, BitmapExpr, FrozenBitmap, FrozenBitmapBuilder};

thread_local! {
    /// Per-thread, because the pools are per-thread and the test harness runs
    /// tests concurrently — a shared counter would tally other tests' work.
    /// `const`-initialised and `Copy`, so reading it never allocates (which
    /// inside a global allocator would recurse).
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn allocs() -> usize {
    ALLOCS.try_with(|c| c.get()).unwrap_or(0)
}

fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

struct Counting;

// SAFETY: every method forwards to `System`, which is a valid allocator; the
// counter is incremented alongside and does not affect the returned pointers.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

/// Allocations performed by `f`, after `warmups` warm-up runs.
///
/// Warm-up must exceed the pool's buffer count: a pool holding buffers grown for
/// a *smaller* op hands them out one per call, so each of the first `buffers`
/// calls of a larger op grows one. Anything past that is steady state.
fn allocs_when_warm(warmups: usize, mut f: impl FnMut()) -> usize {
    for _ in 0..warmups {
        f();
    }
    let before = allocs();
    f();
    allocs() - before
}

#[test]
fn warm_flat_ops_do_not_allocate() {
    let a = build(&(0..20_000).map(|i| i * 3).collect::<Vec<_>>());
    let b = build(&(0..20_000).map(|i| i * 5).collect::<Vec<_>>());
    let v = [a.view(), b.view()];

    let n = allocs_when_warm(24, || {
        std::hint::black_box(intersect_fast(&v));
    });
    assert_eq!(n, 0, "warm intersect_fast allocated {n} times");

    let n = allocs_when_warm(24, || {
        std::hint::black_box(union_fast(&v));
    });
    assert_eq!(n, 0, "warm union_fast allocated {n} times");
}

/// A heap profile over a *varied* workload: many distinct shapes, sizes and
/// container types, so pools are exercised well past their initial warm-up.
/// Reports allocations per iteration; steady state should be flat at zero.
#[test]
fn steady_state_workload_is_allocation_free() {
    let sets: Vec<FrozenBitmap> = (0..8)
        .map(|k| {
            let step = k + 2;
            build(&(0..30_000u32).map(|i| i * step).collect::<Vec<_>>())
        })
        .collect();
    let views: Vec<_> = sets.iter().map(|s| s.view()).collect();

    // One pass over every shape, to warm every pool path.
    let run = |round: usize| {
        for w in 2..=8 {
            let v = &views[..w];
            std::hint::black_box(intersect_fast(v));
            std::hint::black_box(union_fast(v));
            let expr = BitmapExpr::and([
                BitmapExpr::or(v[..w / 2].iter().copied().map(BitmapExpr::leaf)),
                BitmapExpr::leaf(v[w - 1]),
            ]);
            std::hint::black_box(expr.materialize());
        }
        let _ = round;
    };

    run(0);
    run(1);
    let mut per_round = Vec::new();
    for r in 2..8 {
        let before = allocs();
        run(r);
        per_round.push(allocs() - before);
    }
    println!("allocations per steady-state round: {per_round:?}");

    // Tree *construction* still allocates (step lists, shapes, plans) — that is
    // per-query analysis, not per-fold working memory. What must not grow is the
    // fold itself, so the count has to be flat round over round.
    let first = per_round[0];
    assert!(
        per_round.iter().all(|&n| n == first),
        "allocation count is not steady: {per_round:?}"
    );
}

#[test]
fn warm_materialize_does_not_allocate() {
    let a = build(&(0..20_000).map(|i| i * 3).collect::<Vec<_>>());
    let b = build(&(0..20_000).map(|i| i * 5).collect::<Vec<_>>());
    let c = build(&(0..30_000).collect::<Vec<_>>());

    // A tree deep enough to exercise nested arenas, guards, and — since the
    // root AND narrows — the hole-punch mask.
    let expr = BitmapExpr::and([
        BitmapExpr::or([BitmapExpr::leaf(a.view()), BitmapExpr::leaf(b.view())]),
        BitmapExpr::difference(BitmapExpr::leaf(c.view()), BitmapExpr::leaf(b.view())),
        BitmapExpr::leaf(a.view()),
    ]);

    // The mask is built on the first run and cached, so it is part of warm-up.
    let n = allocs_when_warm(24, || {
        std::hint::black_box(expr.materialize());
    });
    assert_eq!(n, 0, "warm materialize allocated {n} times");
}
