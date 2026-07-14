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
    // Partner-major-intersect trial control: if 32-key AND time ≈ 32× the
    // 1-key AND time, key-major intersect has no memory pathology and the
    // partner-major order (which would loosen the capacity clamps) has
    // nothing to win.
    let sparse32 = {
        let mut st = 0x0B5_0F75_u64;
        Set::new(&(0..16).map(|_| arrays(32, 800, &mut st)).collect::<Vec<_>>())
    };
    let fv32 = sparse32.views(16);
    k.bench_function("and16_onekey", |bch| {
        bch.iter(|| black_box(frostbit::ops::kernels::intersect_into(&fv1)))
    });
    k.bench_function("and16_32keys", |bch| {
        bch.iter(|| black_box(frostbit::ops::kernels::intersect_into(&fv32)))
    });
    k.bench_function(format!("fold16_onekey_{RB}"), |bch| {
        bch.iter(|| black_box(rb_diff(&sparse1.rbs[..16])))
    });
    k.finish();

    // Run-container loss cells (vs MultiOps): plan / plan+fold / full split, so
    // the fixed machinery (plan Vecs, arena init, serialize) is separable from
    // the run kernels themselves.
    let mut runs = Set::new(&(0..16).map(|i| run_ranges(16, 4, 6000, i * 1500)).collect::<Vec<_>>());
    runs.optimize_roaring();
    let (rv2, rv16) = (runs.views(2), runs.views(16));
    let mut g = c.benchmark_group("decomp_runs");
    g.bench_function("diff2/plan", |b| b.iter(|| black_box(frostbit::ops::plan::plan_diff(&rv2))));
    g.bench_function("diff2/into", |b| b.iter(|| black_box(frostbit::ops::kernels::diff_into(&rv2))));
    g.bench_function("diff2/full", |b| b.iter(|| black_box(difference_fast(&rv2))));
    g.bench_function(format!("diff2/{RB}"), |b| b.iter(|| black_box(rb_diff(&runs.rbs[..2]))));
    g.bench_function("and16/plan", |b| b.iter(|| black_box(frostbit::ops::plan::plan_intersect(&rv16))));
    g.bench_function("and16/into", |b| b.iter(|| black_box(frostbit::ops::kernels::intersect_into(&rv16))));
    g.bench_function("and16/full", |b| b.iter(|| black_box(intersect_fast(&rv16))));
    g.bench_function(format!("and16/{RB}"), |b| b.iter(|| black_box(rb_and(&runs.rbs[..16]))));
    g.finish();
}
#[cfg(not(feature = "internals"))]
fn decomp(_c: &mut Criterion) {}

// TEMP (profiling): branch-predictability sweep. Same element counts and block
// counts per kernel; only the *pattern* differs — `alt` interleaves the sides
// (advance branch strictly alternates, fully predictable), `rand` scatters
// side membership (advance ~50/50 data-dependent). Overlap sweeps vary the
// intersect/diff hit rate (emit-loop density). Deltas isolate branch stalls.
#[cfg(feature = "internals")]
fn branches(c: &mut Criterion) {
    use frostbit::ops::simd as k;
    const N: usize = 4096;
    let mut st = 0xB4A2_C4E5_u64;

    // Predictable: evens vs odds.
    let a_alt: Vec<u16> = (0..N as u16).map(|i| i * 2).collect();
    let b_alt: Vec<u16> = (0..N as u16).map(|i| i * 2 + 1).collect();

    // Random advance, still disjoint: shuffle 0..2N, first N → a, rest → b.
    let mut idx: Vec<u16> = (0..(2 * N) as u16).collect();
    for i in (1..idx.len()).rev() {
        idx.swap(i, (splitmix64(&mut st) as usize) % (i + 1));
    }
    let (mut a_rnd, mut b_rnd): (Vec<u16>, Vec<u16>) =
        (idx[..N].to_vec(), idx[N..].to_vec());
    a_rnd.sort_unstable();
    b_rnd.sort_unstable();

    // Overlap sweep partners (random advance): p% of b drawn from a.
    let overlap = |p: usize, st: &mut u64| -> Vec<u16> {
        let mut s = std::collections::BTreeSet::new();
        let want_hits = N * p / 100;
        while s.len() < want_hits {
            s.insert(a_rnd[(splitmix64(st) as usize) % N]);
        }
        while s.len() < N {
            let v = (splitmix64(st) % 65536) as u16;
            if a_rnd.binary_search(&v).is_err() {
                s.insert(v);
            }
        }
        s.into_iter().collect()
    };
    let (b_p0, b_p50, b_p100) = (overlap(0, &mut st), overlap(50, &mut st), overlap(100, &mut st));

    let mut out = vec![0u16; 2 * N];
    let mut g = c.benchmark_group("branches");
    for (name, a, b) in [
        ("and/alt", &a_alt, &b_alt),
        ("and/rand", &a_rnd, &b_rnd),
        ("diff/alt", &a_alt, &b_alt),
        ("diff/rand", &a_rnd, &b_rnd),
        ("or/alt", &a_alt, &b_alt),
        ("or/rand", &a_rnd, &b_rnd),
        ("and/hit0", &a_rnd, &b_p0),
        ("and/hit50", &a_rnd, &b_p50),
        ("and/hit100", &a_rnd, &b_p100),
        ("diff/hit50", &a_rnd, &b_p50),
    ] {
        g.bench_function(name, |bch| {
            bch.iter(|| {
                black_box(match name.split('/').next().unwrap() {
                    "and" => k::array_intersect(a, b, &mut out),
                    "diff" => k::array_diff(a, b, &mut out),
                    _ => k::array_union(a, b, &mut out),
                })
            })
        });
    }
    g.finish();
}
#[cfg(not(feature = "internals"))]
fn branches(_c: &mut Criterion) {}

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
    targets = bench, decomp, branches
}
criterion_main!(benches);
