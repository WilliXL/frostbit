//! frostbit vs roaring: N-way fold sweep for AND / OR / DIFF.
//!
//! For each op, arity sweeps 2 → 16 over two operand regimes — sparse arrays
//! and dense bitmaps — so the curves show how each engine scales with fan-in
//! and container type.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use frostbit::{difference_fast, intersect_fast, union_fast, FrozenBitmapView};
use roaring::RoaringBitmap;

#[path = "common.rs"]
mod common;
use common::*;

const ARITIES: [usize; 4] = [2, 4, 8, 16];

fn sweep(
    c: &mut Criterion,
    group: &str,
    fb_op: impl Fn(&[FrozenBitmapView<'_>]) -> frostbit::FrozenBitmap,
    rb_op: impl Fn(&[RoaringBitmap]) -> RoaringBitmap,
    sets: &[(&str, &Set)],
) {
    let mut g = c.benchmark_group(group);
    for (regime, set) in sets {
        for &n in &ARITIES {
            let fv = set.views(n);
            let rv = &set.rbs[..n];
            g.bench_function(format!("{regime}/{n}/frostbit"), |b| b.iter(|| black_box(fb_op(&fv))));
            g.bench_function(format!("{regime}/{n}/roaring"), |b| b.iter(|| black_box(rb_op(rv))));
        }
    }
    g.finish();
}

fn bench(c: &mut Criterion) {
    let mut st = 0x0B5_0F75_u64;
    // 16 sparse-array inputs and 16 dense-bitmap inputs (phase-shifted to keep
    // intersections non-empty and unions non-trivial).
    let sparse = Set::new(&(0..16).map(|_| arrays(32, 800, &mut st)).collect::<Vec<_>>());
    let dense = Set::new(&(0..16).map(|i| dense(16, 5000, i * 97, &mut st)).collect::<Vec<_>>());
    // Dense run containers (a few long ranges per key) — exercises the native
    // run path instead of bitmap expansion. Roaring is run-optimized here so
    // it's run-vs-run (best-vs-best), not run-vs-bitmap.
    let mut runs = Set::new(&(0..16).map(|i| run_ranges(16, 4, 6000, i * 1500)).collect::<Vec<_>>());
    runs.optimize_roaring();

    // Cross-engine parity over the full 16-way fold of each op.
    for (name, set) in [("sparse", &sparse), ("dense", &dense), ("runs", &runs)] {
        let fv = set.views(set.len());
        let rv = &set.rbs[..];
        assert_eq!(fb_vec(&intersect_fast(&fv)), rb_vec(&rb_and(rv)), "AND {name}");
        assert_eq!(fb_vec(&union_fast(&fv)), rb_vec(&rb_or(rv)), "OR {name}");
        assert_eq!(fb_vec(&difference_fast(&fv)), rb_vec(&rb_diff(rv)), "DIFF {name}");
    }

    let sets: [(&str, &Set); 3] = [("sparse", &sparse), ("dense", &dense), ("runs", &runs)];
    sweep(c, "intersect", |v| intersect_fast(v), rb_and, &sets);
    sweep(c, "union", |v| union_fast(v), rb_or, &sets);
    sweep(c, "diff", |v| difference_fast(v), rb_diff, &sets);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1200))
        .measurement_time(Duration::from_secs(4));
    targets = bench
}
criterion_main!(benches);
