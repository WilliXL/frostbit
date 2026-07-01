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
            g.bench_function(format!("{regime}/{n}/{RB}"), |b| b.iter(|| black_box(rb_op(rv))));
        }
    }
    g.finish();
}

// TEMP (profiling): decompose diff/sparse into plan / fold / full per arity.
#[cfg(feature = "internals")]
fn decomp(c: &mut Criterion) {
    let mut st = 0x0B5_0F75_u64;
    let sparse = Set::new(&(0..16).map(|_| arrays(32, 800, &mut st)).collect::<Vec<_>>());
    let mut g = c.benchmark_group("decomp");
    for n in [2usize, 4, 8, 16] {
        let fv = sparse.views(n);
        g.bench_function(format!("plan/{n}"), |b| {
            b.iter(|| black_box(frostbit::ops::plan::plan_diff(&fv)))
        });
        g.bench_function(format!("fold/{n}"), |b| {
            b.iter(|| black_box(frostbit::ops::kernels::diff_into(&fv)))
        });
        g.bench_function(format!("full/{n}"), |b| b.iter(|| black_box(difference_fast(&fv))));
        g.bench_function(format!("{RB}/{n}"), |b| {
            b.iter(|| black_box(rb_diff(&sparse.rbs[..n])))
        });
    }
    g.finish();

    // Standalone kernel throughput at fold shapes (blocks = (na+nb)/8).
    let mut st = 0xFEED_BEEF_u64;
    let gen_arr = |n: usize, st: &mut u64| -> Vec<u16> {
        let mut s = std::collections::BTreeSet::new();
        while s.len() < n {
            s.insert((splitmix64(st) % 65536) as u16);
        }
        s.into_iter().collect()
    };
    let (a800, b800, a660) = (gen_arr(800, &mut st), gen_arr(800, &mut st), gen_arr(660, &mut st));
    let mut out = vec![0u16; 4096];
    let mut k = c.benchmark_group("kernel");
    k.bench_function("diff/800x800", |bch| {
        bch.iter(|| black_box(frostbit::ops::simd::array_diff(&a800, &b800, &mut out)))
    });
    k.bench_function("diff/660x800", |bch| {
        bch.iter(|| black_box(frostbit::ops::simd::array_diff(&a660, &b800, &mut out)))
    });
    k.bench_function("intersect/800x800", |bch| {
        bch.iter(|| black_box(frostbit::ops::simd::array_intersect(&a800, &b800, &mut out)))
    });
    k.bench_function("union/800x800", |bch| {
        bch.iter(|| black_box(frostbit::ops::simd::array_union(&a800, &b800, &mut out)))
    });

    // Same 16-way fold, but a single key (~26 KB working set, L1-resident):
    // isolates the fold's memory access pattern from its instruction stream.
    let one_key: Vec<Vec<u32>> = (0..16)
        .map(|_| {
            let mut s = std::collections::BTreeSet::new();
            while s.len() < 800 {
                s.insert((splitmix64(&mut st) % 65536) as u32);
            }
            s.into_iter().collect()
        })
        .collect();
    let sparse1 = Set::new(&one_key);
    let fv1 = sparse1.views(16);
    k.bench_function("fold16_onekey", |bch| {
        bch.iter(|| black_box(frostbit::ops::kernels::diff_into(&fv1)))
    });
    k.bench_function(format!("fold16_onekey_{RB}"), |bch| {
        bch.iter(|| black_box(rb_diff(&sparse1.rbs[..16])))
    });
    k.finish();
}
#[cfg(not(feature = "internals"))]
fn decomp(_c: &mut Criterion) {}

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
    targets = bench, decomp
}
criterion_main!(benches);
