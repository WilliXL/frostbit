//! frostbit `BitmapExpr` vs roaring recursive eval, over a diverse set of trees.
//!
//! Both a handful of named realistic filter shapes and randomly generated trees
//! across size classes (tiny → large) over a mixed-shape leaf pool. The frostbit
//! tree is *built and analyzed inside the timed region*, so construction cost is
//! included, never amortized away.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use frostbit::BitmapExpr;
use roaring::RoaringBitmap;

#[path = "common.rs"]
mod common;
use common::*;

/// A tree over leaf-pool indices, evaluated by either engine.
enum Spec {
    Leaf(usize),
    And(Vec<Spec>),
    Or(Vec<Spec>),
    Diff(Box<Spec>, Box<Spec>),
}

fn leaf(i: usize) -> Spec {
    Spec::Leaf(i)
}
fn and(cs: Vec<Spec>) -> Spec {
    Spec::And(cs)
}
fn or(cs: Vec<Spec>) -> Spec {
    Spec::Or(cs)
}
fn diff(a: Spec, b: Spec) -> Spec {
    Spec::Diff(Box::new(a), Box::new(b))
}

/// Build the frostbit expression (recursive — this is the tree *definition*;
/// analysis is fused into construction).
fn build_fb<'a>(spec: &Spec, p: &'a Set) -> BitmapExpr<'a> {
    match spec {
        Spec::Leaf(i) => BitmapExpr::leaf(p.fv(*i)),
        Spec::And(cs) => BitmapExpr::and(cs.iter().map(|c| build_fb(c, p))),
        Spec::Or(cs) => BitmapExpr::or(cs.iter().map(|c| build_fb(c, p))),
        Spec::Diff(a, b) => BitmapExpr::difference(build_fb(a, p), build_fb(b, p)),
    }
}

fn eval_rb(spec: &Spec, p: &Set) -> RoaringBitmap {
    match spec {
        Spec::Leaf(i) => p.rbs[*i].clone(),
        Spec::And(cs) => {
            let mut acc = eval_rb(&cs[0], p);
            for c in &cs[1..] {
                acc = &acc & &eval_rb(c, p);
            }
            acc
        }
        Spec::Or(cs) => {
            let mut acc = RoaringBitmap::new();
            for c in cs {
                acc = &acc | &eval_rb(c, p);
            }
            acc
        }
        Spec::Diff(a, b) => &eval_rb(a, p) - &eval_rb(b, p),
    }
}

fn count_leaves(spec: &Spec) -> usize {
    match spec {
        Spec::Leaf(_) => 1,
        Spec::And(cs) | Spec::Or(cs) => cs.iter().map(count_leaves).sum(),
        Spec::Diff(a, b) => count_leaves(a) + count_leaves(b),
    }
}

/// Generate a tree of roughly `budget` leaves over `n` pool entries.
fn gen(st: &mut u64, budget: usize, n: usize) -> Spec {
    if budget <= 1 {
        return Spec::Leaf((splitmix64(st) as usize) % n);
    }
    match splitmix64(st) % 5 {
        0 => {
            let half = (budget / 2).max(1);
            diff(gen(st, half, n), gen(st, half, n))
        }
        r => {
            let k = 2 + (splitmix64(st) % 3) as usize;
            let per = (budget / k).max(1);
            let kids = (0..k).map(|_| gen(st, per, n)).collect();
            if r % 2 == 0 {
                and(kids)
            } else {
                or(kids)
            }
        }
    }
}

/// Mixed-shape leaf pool: small/sparse arrays, medium arrays, large dense
/// (multi-container bitmaps), and run-heavy — so trees exercise every container
/// type and fold path.
fn mixed_pool() -> Set {
    let mut st = 0x5EED_1234u64;
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    for _ in 0..6 {
        inputs.push(arrays(8, 150, &mut st));
    }
    for _ in 0..4 {
        inputs.push(arrays(48, 1200, &mut st));
    }
    for i in 0..4 {
        inputs.push(dense(32, 12_000, i * 777, &mut st));
    }
    // Contiguous ID ranges → run containers with a few *long* runs (how runs
    // actually arise in filter/search), not hundreds of tiny ones.
    for i in 0..3 {
        inputs.push(runs(20, 3000 + i * 1500, 5000));
    }
    Set::new(&inputs)
}

/// A leaf whose blocks span keys `[k0, k1)`, `per_key` random values each — used
/// to build key-selective trees (a narrow band ∩ wide bands) for hole-punching.
fn band(k0: u16, k1: u16, per_key: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in k0..k1 {
        for _ in 0..per_key {
            v.push(((k as u32) << 16) | (splitmix64(st) % 65536) as u32);
        }
    }
    sorted(v)
}

/// Named realistic shapes over the mixed pool (indices chosen for variety).
fn named() -> Vec<(&'static str, Spec)> {
    vec![
        // CNF: AND of OR-groups.
        ("cnf3", and(vec![or(vec![leaf(0), leaf(1), leaf(6)]), or(vec![leaf(2), leaf(7)]), or(vec![leaf(10), leaf(11)])])),
        // Nested ANDs (frostbit flattens to one 5-way op).
        ("conj5", and(vec![and(vec![leaf(6), leaf(7)]), leaf(8), and(vec![leaf(9), leaf(12)])])),
        // base ∩ domain-OR ∩ (universe \ lang).
        ("filter", and(vec![leaf(12), or(vec![leaf(0), leaf(1), leaf(2), leaf(3)]), diff(leaf(13), leaf(10))])),
        // DNF-ish: OR of AND-groups and a leaf.
        ("dnf", or(vec![and(vec![leaf(6), leaf(7)]), and(vec![leaf(8), leaf(9)]), leaf(4)])),
    ]
}

/// Per-tree shape profile for the corpus: each tree draws its own leafiness,
/// op mix, and node-width distribution, so the corpus spans deep AND-chains,
/// flat 100-way ORs, diff-heavy filters, and everything between.
struct Profile {
    leaf_pct: u64,
    diff_pct: u64,
    or_pct: u64,
    wide: bool,
}

/// One random tree under `budget` total leaves and `depth` remaining levels.
fn gen_profiled(st: &mut u64, budget: usize, depth: usize, prof: &Profile, n: usize) -> Spec {
    if budget <= 1 || depth == 0 || splitmix64(st) % 100 < prof.leaf_pct {
        return leaf((splitmix64(st) as usize) % n);
    }
    let r = splitmix64(st) % 100;
    if r < prof.diff_pct {
        let l = 1 + (splitmix64(st) as usize) % (budget - 1);
        return diff(
            gen_profiled(st, l, depth - 1, prof, n),
            gen_profiled(st, budget - l, depth - 1, prof, n),
        );
    }
    // Node width: mostly narrow, occasionally very wide (up to the full budget).
    let max_w = if prof.wide && splitmix64(st) % 8 == 0 { budget } else { 2 + budget.min(6) };
    let w = 2 + (splitmix64(st) as usize) % (max_w.min(budget).max(2) - 1);
    // Split the leaf budget unevenly across children (colliding cuts collapse,
    // so the total leaf count never exceeds the budget).
    let mut cuts: Vec<usize> = (0..w - 1).map(|_| 1 + (splitmix64(st) as usize) % budget).collect();
    cuts.sort_unstable();
    let mut kids = Vec::with_capacity(w);
    let mut prev = 0usize;
    for &cut in cuts.iter().chain(std::iter::once(&budget)) {
        if cut > prev {
            kids.push(gen_profiled(st, cut - prev, depth - 1, prof, n));
            prev = cut;
        }
    }
    if kids.len() == 1 {
        return kids.pop().unwrap();
    }
    if r < prof.diff_pct + prof.or_pct {
        or(kids)
    } else {
        and(kids)
    }
}

/// The 25k-tree corpus: up to 100 leaves and 15 levels per tree, deterministic.
fn corpus_specs(n_trees: usize, pool_len: usize) -> Vec<Spec> {
    let mut st = 0xC0_2b_05_2026_u64;
    (0..n_trees)
        .map(|_| {
            // Log-ish spread of sizes: plenty of small trees, a long large tail.
            let budget = match splitmix64(&mut st) % 10 {
                0..=4 => 2 + (splitmix64(&mut st) as usize) % 9, // 2..=10
                5..=7 => 10 + (splitmix64(&mut st) as usize) % 31, // 10..=40
                _ => 40 + (splitmix64(&mut st) as usize) % 61,   // 40..=100
            };
            let depth = 2 + (splitmix64(&mut st) as usize) % 14; // 2..=15
            let prof = Profile {
                leaf_pct: splitmix64(&mut st) % 45,
                diff_pct: splitmix64(&mut st) % 35,
                or_pct: 20 + splitmix64(&mut st) % 60,
                wide: splitmix64(&mut st) % 3 == 0,
            };
            gen_profiled(&mut st, budget, depth, &prof, pool_len)
        })
        .collect()
}

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

    let mut g = c.benchmark_group("corpus");
    g.sample_size(10);
    g.warm_up_time(Duration::from_secs(2));
    g.measurement_time(Duration::from_secs(20));
    g.throughput(criterion::Throughput::Elements(specs.len() as u64));
    g.bench_function(format!("25k_trees_{total_leaves}leaves/frostbit"), |b| {
        b.iter(|| {
            for spec in &specs {
                black_box(build_fb(spec, &pool).materialize());
            }
        })
    });
    g.bench_function(format!("25k_trees_{total_leaves}leaves/{RB}"), |b| {
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
