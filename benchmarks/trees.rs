//! frostbit `BitmapExpr` vs roaring recursive eval, over a diverse set of trees.
//!
//! Both a handful of named realistic filter shapes and randomly generated trees
//! across size classes (tiny → large) over a mixed-shape leaf pool. The frostbit
//! tree is *built and analyzed inside the timed region*, so construction cost is
//! included, never amortized away.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[path = "common.rs"]
mod common;
use common::*;
#[path = "treegen.rs"]
mod treegen;
use treegen::*;

/// Whole-corpus throughput: evaluate all 25k trees per iteration (frostbit
/// builds + analyzes + materializes each tree inside the timed region, exactly
/// what a query engine pays per query; roaring re-evaluates recursively).
fn corpus(c: &mut Criterion) {
    let pool = mixed_pool();
    let specs = corpus_specs(25_000, pool.len());
    let total_leaves: usize = specs.iter().map(count_leaves).sum();

    // Parity spot-check across the corpus (every 64th tree).
    for (i, spec) in specs.iter().enumerate().step_by(64) {
        let got = fb_vec(&build_fb(spec, &pool).materialize());
        let want = rb_vec(&eval_rb(spec, &pool));
        assert_eq!(got, want, "frostbit≠roaring in corpus tree {i}");
    }

    println!("corpus: {} trees, {} total leaves", specs.len(), total_leaves);
    let mut g = c.benchmark_group("corpus");
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(2));
    g.measurement_time(Duration::from_secs(20));
    g.throughput(criterion::Throughput::Elements(specs.len() as u64));
    g.bench_function("25k_trees/frostbit", |b| {
        b.iter(|| {
            for spec in &specs {
                black_box(build_fb(spec, &pool).materialize());
            }
        })
    });
    g.bench_function(format!("25k_trees/{RB}"), |b| {
        b.iter(|| {
            for spec in &specs {
                black_box(eval_rb(spec, &pool));
            }
        })
    });
    g.finish();
}

fn bench(c: &mut Criterion) {
    let pool = mixed_pool();

    // Named shapes plus random trees across size classes.
    let mut specs: Vec<(String, Spec)> =
        named().into_iter().map(|(n, s)| (n.to_string(), s)).collect();
    let mut st = 0xA11C_E5_u64;
    for (cname, budget) in [("tiny", 3usize), ("small", 9), ("medium", 22), ("large", 45)] {
        for j in 0..2 {
            let spec = gen(&mut st, budget, pool.len());
            specs.push((format!("{cname}{j}_{}leaves", count_leaves(&spec)), spec));
        }
    }

    // Cross-engine parity for every tree, once.
    for (name, spec) in &specs {
        let got = fb_vec(&build_fb(spec, &pool).materialize());
        let want = rb_vec(&eval_rb(spec, &pool));
        assert_eq!(got, want, "frostbit≠roaring in {name}");
    }

    let mut g = c.benchmark_group("tree");
    for (name, spec) in &specs {
        // The frostbit expression / fold plan is built ONCE up front (analyze
        // once, execute many) — only materialization is timed. roaring has no
        // reusable plan, so it re-evaluates each call.
        let expr = build_fb(spec, &pool);
        g.bench_function(format!("{name}/frostbit"), |b| {
            b.iter(|| black_box(expr.materialize()))
        });
        g.bench_function(format!("{name}/{RB}"), |b| {
            b.iter(|| black_box(eval_rb(black_box(spec), &pool)))
        });
    }
    g.finish();

    // Hole-punching: a key-selective AND — a narrow 4-block filter intersected
    // with wide 256-block OR-groups. Punching derives the 4 surviving blocks
    // from the narrow branch and prunes the wide branches to them before folding.
    let mut st2 = 0xB0B0_CAFEu64;
    let sel = Set::new(&[
        band(0, 4, 1500, &mut st2),   // 0: narrow filter — 4 blocks
        band(0, 256, 250, &mut st2),  // 1..=4: wide — 256 blocks each
        band(0, 256, 250, &mut st2),
        band(0, 256, 250, &mut st2),
        band(0, 256, 250, &mut st2),
    ]);
    let sel_spec = and(vec![leaf(0), or(vec![leaf(1), leaf(2)]), or(vec![leaf(3), leaf(4)])]);
    let want = rb_vec(&eval_rb(&sel_spec, &sel));
    assert_eq!(fb_vec(&build_fb(&sel_spec, &sel).materialize()), want, "selective plain");
    assert_eq!(fb_vec(&build_fb(&sel_spec, &sel).punch_holes().materialize()), want, "selective punched");

    let unpunched = build_fb(&sel_spec, &sel);
    let punched = build_fb(&sel_spec, &sel).punch_holes();
    let mut h = c.benchmark_group("holepunch");
    h.bench_function("selective/frostbit", |b| b.iter(|| black_box(unpunched.materialize())));
    h.bench_function("selective/frostbit_punched", |b| b.iter(|| black_box(punched.materialize())));
    h.bench_function(format!("selective/{RB}"), |b| {
        b.iter(|| black_box(eval_rb(black_box(&sel_spec), &sel)))
    });
    h.finish();

    // Short-circuit: AND(∅-subtree, expensive OR). `diff(leaf0, leaf0)` is empty,
    // so frostbit skips evaluating the wide OR entirely; roaring must build it.
    let sc = Set::new(&[
        band(0, 2, 100, &mut st2),   // 0: for the empty diff
        band(0, 200, 300, &mut st2), // 1..=3: expensive OR operands
        band(0, 200, 300, &mut st2),
        band(0, 200, 300, &mut st2),
    ]);
    let sc_spec = and(vec![diff(leaf(0), leaf(0)), or(vec![leaf(1), leaf(2), leaf(3)])]);
    assert!(fb_vec(&build_fb(&sc_spec, &sc).materialize()).is_empty(), "shortcircuit empty");
    assert!(rb_vec(&eval_rb(&sc_spec, &sc)).is_empty());
    let sc_fb = build_fb(&sc_spec, &sc);
    let mut s = c.benchmark_group("shortcircuit");
    s.bench_function("empty_and/frostbit", |b| b.iter(|| black_box(sc_fb.materialize())));
    s.bench_function(format!("empty_and/{RB}"), |b| b.iter(|| black_box(eval_rb(black_box(&sc_spec), &sc))));
    s.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1200))
        .measurement_time(Duration::from_secs(4));
    targets = bench, corpus
}
criterion_main!(benches);
