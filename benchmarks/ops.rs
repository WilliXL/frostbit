//! frostbit vs roaring: N-way fold sweep for AND / OR / DIFF.
//!
//! The cross-engine comparison. White-box measurements of frostbit's own
//! internals live in `micro.rs`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use frostbit::{difference_fast, intersect_fast, union_fast, FrozenBitmapView};
use roaring::RoaringBitmap;
use std::time::Duration;

#[path = "support/common.rs"]
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
    sweep(c, "intersect", intersect_fast, rb_and, &sets);
    sweep(c, "union", union_fast, rb_or, &sets);
    sweep(c, "diff", difference_fast, rb_diff, &sets);
}

/// Fan-in for the coverage group. The sweep above owns the arity curve; here
/// one arity is enough, and 8 is wide enough that a fold reaches steady state.
const COV_N: usize = 8;

/// The conditions the homogeneous sweep cannot reach.
///
/// Every regime above is one container form, over one key range, at one
/// cardinality — so three whole classes of path never run: the cross-form
/// kernels (array∩bitmap, run∩bitmap, …), key structures that change the merge
/// itself rather than the per-key work (disjoint, nested, lopsided), and the
/// cardinalities where the analyzer's form decision flips. That last class is
/// where the parent-slot bug lived, and no hand-written shape had reached it.
///
/// Each regime is asserted against roaring on all three ops before it is timed,
/// so this group is a correctness net as much as a performance one.
fn coverage(c: &mut Criterion) {
    let mut st = 0xC0_5E_2026_u64;
    let n = COV_N as u16;

    // Inline (FI) operands: 4000 keys holding one value each, so 4 bytes/value
    // beats a container header and the builder picks inline.
    let inline = Set::new(&(0..n).map(|i| thin(4000, 1, i as u32 * 7919, &mut st)).collect::<Vec<_>>());
    // Alternating inline and array operands — the cross-format seed paths.
    let inline_array = Set::new(
        &(0..n)
            .map(|i| {
                if i % 2 == 0 { thin(512, 2, i as u32 * 7919, &mut st) } else { arrays(512, 900, &mut st) }
            })
            .collect::<Vec<_>>(),
    );
    // One operand of each form over a shared key range: every pairing of
    // array/bitmap/run meets in the fold, which homogeneous regimes never do.
    let mixed = Set::new(
        &(0..n)
            .map(|i| match i % 3 {
                0 => arrays(64, 900, &mut st),
                1 => dense(64, 20_000, i as u32 * 97, &mut st),
                _ => run_ranges(64, 4, 6000, i as u32 * 1500),
            })
            .collect::<Vec<_>>(),
    );
    // Key structure, not per-key content: no shared keys at all (AND resolves
    // in the key merge; OR never opens a container), …
    let disjoint =
        Set::new(&(0..n).map(|i| key_band(i * 64, 64, 900, &mut st)).collect::<Vec<_>>());
    // … each operand's keys a subset of the one before, …
    let nested = Set::new(
        &(0..n).map(|i| key_band(0, 256 >> i.min(5), 900, &mut st)).collect::<Vec<_>>(),
    );
    // … and one narrow operand against wide ones, so the key merge spends its
    // time skipping rather than stepping.
    let asymmetric = Set::new(
        &(0..n)
            .map(|i| if i == 0 { key_band(0, 2, 900, &mut st) } else { key_band(0, 512, 900, &mut st) })
            .collect::<Vec<_>>(),
    );
    // Cardinality astride ARRAY_MAX_SIZE (4096) and run count astride MAX_RUNS
    // (2047): the two thresholds the analyzer's form decision turns on, sampled
    // just under and just over so both sides of each branch are timed.
    let arr_under = Set::new(&(0..n).map(|_| arrays(64, 4090, &mut st)).collect::<Vec<_>>());
    let arr_over = Set::new(&(0..n).map(|_| arrays(64, 4100, &mut st)).collect::<Vec<_>>());
    let mut run_under = Set::new(&(0..n).map(|i| run_count(16, 2040, i as u32)).collect::<Vec<_>>());
    let mut run_over = Set::new(&(0..n).map(|i| run_count(16, 2060, i as u32)).collect::<Vec<_>>());
    // Saturated containers: every kernel's output is its input.
    let mut full = Set::new(&(0..n).map(|_| full_keys(0, 16)).collect::<Vec<_>>());
    for s in [&mut run_under, &mut run_over, &mut full] {
        s.optimize_roaring(); // run-vs-run, not run-vs-bitmap
    }

    let regimes: [(&str, &Set); 11] = [
        ("inline", &inline),
        ("inline_array", &inline_array),
        ("mixed_forms", &mixed),
        ("disjoint", &disjoint),
        ("nested", &nested),
        ("asymmetric", &asymmetric),
        ("array_edge_under", &arr_under),
        ("array_edge_over", &arr_over),
        ("run_edge_under", &run_under),
        ("run_edge_over", &run_over),
        ("full", &full),
    ];

    for (name, set) in regimes {
        let fv = set.views(COV_N);
        let rv = &set.rbs[..COV_N];
        assert_eq!(fb_vec(&intersect_fast(&fv)), rb_vec(&rb_and(rv)), "AND {name}");
        assert_eq!(fb_vec(&union_fast(&fv)), rb_vec(&rb_or(rv)), "OR {name}");
        assert_eq!(fb_vec(&difference_fast(&fv)), rb_vec(&rb_diff(rv)), "DIFF {name}");
    }

    let mut g = c.benchmark_group("coverage");
    for (name, set) in regimes {
        let fv = set.views(COV_N);
        let rv = &set.rbs[..COV_N];
        for (op, fb_op, rb_op) in [
            ("intersect", intersect_fast as fn(&[FrozenBitmapView<'_>]) -> _, rb_and as fn(&[RoaringBitmap]) -> _),
            ("union", union_fast, rb_or),
            ("diff", difference_fast, rb_diff),
        ] {
            g.bench_function(format!("{name}/{op}/frostbit"), |b| b.iter(|| black_box(fb_op(&fv))));
            g.bench_function(format!("{name}/{op}/{RB}"), |b| b.iter(|| black_box(rb_op(rv))));
        }
    }
    g.finish();
}


criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1200))
        .measurement_time(Duration::from_secs(4));
    targets = bench, coverage
}
criterion_main!(benches);
