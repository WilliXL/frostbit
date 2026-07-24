//! Working-memory pool: budgeting, pre-allocation, stats, and overflow policy.
//!
//! Every test runs on its own thread because the pools (and their budget) are
//! per-thread — that isolation is itself part of the contract.

use frostbit::pool::{self, OnOverflow, PoolConfig, PoolStats};
use frostbit::{intersect_fast, union_fast, BitmapExpr, FrozenBitmap, FrozenBitmapBuilder};

fn build(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

/// Run `f` on a fresh thread, so its pool state and budget are its own.
fn on_own_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f).join().expect("test thread panicked")
}

#[test]
fn config_round_trips() {
    on_own_thread(|| {
        pool::configure(PoolConfig::new().buffers(4).buffer_bytes(1 << 16));
        let c = pool::config();
        assert_eq!(c, PoolConfig::new().buffers(4).buffer_bytes(1 << 16));
        // Both byte pools are warmed (they swap buffers), hence the 2x.
        assert_eq!(c.prewarm_bytes(), 2 * 4 * (1 << 16));

        // An explicit shape sets both the sizes and the count.
        let shaped = PoolConfig::new().buffer_sizes([8, 16, 32]);
        assert_eq!(shaped.prewarm_bytes(), 2 * 56);
        pool::configure(shaped.clone());
        assert_eq!(pool::config(), shaped);
    });
}

#[test]
fn prewarm_allocates_the_budget_up_front() {
    on_own_thread(|| {
        let before = pool::stats();
        assert_eq!(before.retained_bytes, 0, "fresh thread holds nothing");

        pool::configure(PoolConfig::new().buffers(3).buffer_bytes(64 << 10));
        pool::prewarm();

        let after = pool::stats();
        // Both byte pools (arena working memory + result buffers) are warmed.
        assert_eq!(after.retained, 6, "3 buffers in each of the 2 byte pools");
        assert!(
            after.retained_bytes >= 6 * (64 << 10),
            "expected >= 384 KiB reserved, got {}",
            after.retained_bytes
        );
        assert_eq!(after.live, 0, "nothing is handed out yet");
    });
}

#[test]
fn clear_releases_retained_memory() {
    on_own_thread(|| {
        pool::configure(PoolConfig::new().buffers(2).buffer_bytes(32 << 10));
        pool::prewarm();
        assert!(pool::stats().retained_bytes > 0);

        pool::clear();
        let s = pool::stats();
        assert_eq!((s.retained, s.retained_bytes), (0, 0));
    });
}

#[test]
fn budget_bounds_retained_buffers() {
    on_own_thread(|| {
        pool::configure(PoolConfig::new().buffers(2));
        // Many sequential ops: each returns its buffer, but the pool retains at
        // most the budget.
        let a = build(&(0..5_000).map(|i| i * 2).collect::<Vec<_>>());
        let b = build(&(0..5_000).map(|i| i * 3).collect::<Vec<_>>());
        for _ in 0..20 {
            let _ = intersect_fast(&[a.view(), b.view()]);
            let _ = union_fast(&[a.view(), b.view()]);
        }
        let s = pool::stats();
        assert!(s.retained <= 4, "2 byte pools x budget 2, got {}", s.retained);
        // `a` and `b` are still alive and each owns its result buffer; every
        // op-scoped buffer was returned.
        assert_eq!(s.live, 2, "only the two live bitmaps are still holding");
        drop((a, b));
        assert_eq!(pool::stats().live, 0, "dropping the bitmaps returns them");
    });
}

#[test]
fn overflow_allocates_by_default_and_results_stay_correct() {
    on_own_thread(|| {
        // A budget of 1 is far below what a deep tree needs live at once.
        pool::configure(PoolConfig::new().buffers(1));
        let a = build(&(0..4_000).collect::<Vec<_>>());
        let b = build(&(0..4_000).map(|i| i * 2).collect::<Vec<_>>());
        let c = build(&(1_000..6_000).collect::<Vec<_>>());

        let expr = BitmapExpr::and([
            BitmapExpr::or([BitmapExpr::leaf(a.view()), BitmapExpr::leaf(b.view())]),
            BitmapExpr::difference(BitmapExpr::leaf(c.view()), BitmapExpr::leaf(b.view())),
        ]);
        let got: Vec<u32> = expr.materialize().iter().collect();

        // Oracle: (a ∪ b) ∩ (c \ b)
        use std::collections::BTreeSet;
        let (sa, sb): (BTreeSet<u32>, BTreeSet<u32>) =
            (a.iter().collect(), b.iter().collect());
        let sc: BTreeSet<u32> = c.iter().collect();
        let want: Vec<u32> = sa
            .union(&sb)
            .copied()
            .filter(|v| sc.contains(v) && !sb.contains(v))
            .collect();
        assert_eq!(got, want, "an over-budget fold must still be correct");

        let PoolStats { overflows, live, .. } = pool::stats();
        assert!(overflows > 0, "a budget of 1 should have overflowed");
        // Only the three inputs (each owning its result buffer) remain: the
        // over-budget temporaries were dropped, not leaked or retained.
        assert_eq!(live, 3, "fold temporaries are released");
        drop((a, b, c));
        assert_eq!(pool::stats().live, 0);
    });
}

#[test]
fn overflow_fail_panics_with_a_named_budget() {
    let panicked = std::thread::spawn(|| {
        pool::configure(PoolConfig::new().buffers(1).on_overflow(OnOverflow::Fail));
        let a = build(&(0..4_000).collect::<Vec<_>>());
        let b = build(&(0..4_000).map(|i| i * 2).collect::<Vec<_>>());
        // Nested folds need more than one buffer live at once.
        let expr = BitmapExpr::and([
            BitmapExpr::or([BitmapExpr::leaf(a.view()), BitmapExpr::leaf(b.view())]),
            BitmapExpr::leaf(a.view()),
        ]);
        let _ = expr.materialize();
    })
    .join();
    let err = panicked.expect_err("OnOverflow::Fail must panic");
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_default();
    assert!(msg.contains("budget exceeded"), "unhelpful panic: {msg:?}");
}

#[test]
fn budgets_are_per_thread() {
    on_own_thread(|| {
        pool::configure(PoolConfig::new().buffers(5).buffer_bytes(1 << 10));
        pool::prewarm();
        assert!(pool::stats().retained > 0);

        // A different thread starts from the default budget, holding nothing.
        let other = on_own_thread(|| (pool::stats(), pool::config()));
        assert_eq!(other.0.retained, 0);
        assert_eq!(other.1, PoolConfig::new());
    });
}
