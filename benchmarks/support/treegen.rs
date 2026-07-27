//! Shared tree machinery for the tree benches and the per-tree audit:
//! the `Spec` definition, both engines' evaluators, the leaf pools, the named
//! shapes, and the (deterministic) 25k-tree corpus generator.
#![allow(dead_code)]

use frostbit::BitmapExpr;
use roaring::{MultiOps, RoaringBitmap};

use crate::common::*;

/// A tree over leaf-pool indices, evaluated by either engine.
pub enum Spec {
    Leaf(usize),
    And(Vec<Spec>),
    Or(Vec<Spec>),
    Diff(Box<Spec>, Box<Spec>),
}

pub fn leaf(i: usize) -> Spec {
    Spec::Leaf(i)
}
pub fn and(cs: Vec<Spec>) -> Spec {
    Spec::And(cs)
}
pub fn or(cs: Vec<Spec>) -> Spec {
    Spec::Or(cs)
}
pub fn diff(a: Spec, b: Spec) -> Spec {
    Spec::Diff(Box::new(a), Box::new(b))
}

/// Build the frostbit expression (recursive — this is the tree *definition*;
/// analysis is fused into construction).
pub fn build_fb<'a>(spec: &Spec, p: &'a Set) -> BitmapExpr<'a> {
    match spec {
        Spec::Leaf(i) => BitmapExpr::leaf(p.fv(*i)),
        Spec::And(cs) => BitmapExpr::and(cs.iter().map(|c| build_fb(c, p))),
        Spec::Or(cs) => BitmapExpr::or(cs.iter().map(|c| build_fb(c, p))),
        Spec::Diff(a, b) => BitmapExpr::difference(build_fb(a, p), build_fb(b, p)),
    }
}

/// N-ary groups go through `MultiOps` (the library's documented fast path);
/// binary diff stays the pairwise operator.
pub fn eval_rb(spec: &Spec, p: &Set) -> RoaringBitmap {
    match spec {
        Spec::Leaf(i) => p.rbs[*i].clone(),
        Spec::And(cs) => cs.iter().map(|c| eval_rb(c, p)).intersection(),
        Spec::Or(cs) => cs.iter().map(|c| eval_rb(c, p)).union(),
        Spec::Diff(a, b) => &eval_rb(a, p) - &eval_rb(b, p),
    }
}

pub fn count_leaves(spec: &Spec) -> usize {
    match spec {
        Spec::Leaf(_) => 1,
        Spec::And(cs) | Spec::Or(cs) => cs.iter().map(count_leaves).sum(),
        Spec::Diff(a, b) => count_leaves(a) + count_leaves(b),
    }
}

/// Generate a tree of roughly `budget` leaves over `n` pool entries.
pub fn gen(st: &mut u64, budget: usize, n: usize) -> Spec {
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
pub fn mixed_pool() -> Set {
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
    // --- classes the original pool had none of (indices 17.. are additive, so
    // the named shapes above keep their leaf indices) ---
    // Tiny scattered sets: small enough that the builder picks the inline (FI)
    // encoding, which nothing else in the pool exercises.
    for i in 0..2 {
        inputs.push(sorted((0..40u32).map(|v| v * 997 + i * 13).collect()));
    }
    // A single dense container, and one that fills a whole 64K block: the
    // boundaries where array/bitmap/run selection flips.
    inputs.push((0..9000u32).collect());
    inputs.push((0..65_536u32).collect());
    // Disjoint key bands — an AND over these is statically empty, which is the
    // planner's early-out, and an OR over them never overlaps.
    for i in 0..3u32 {
        let base = (600 + i * 200) << 16;
        inputs.push(sorted((0..4000u32).map(|v| base + v * 7).collect()));
    }
    // A dense *run* leaf and a *bitmap* leaf over the SAME keys — the shape
    // BUG-3 needed (run accumulator meeting a bitmap subtrahend at one key).
    inputs.push(sorted((0..12u32).flat_map(|k| (0..20_000u32).map(move |v| (k << 16) | v)).collect()));
    inputs.push(sorted((0..12u32).flat_map(|k| (0..30_000u32).map(move |v| (k << 16) | (v * 2))).collect()));
    // Extreme skew: a handful of values that a huge leaf mostly contains.
    inputs.push(sorted((0..24u32).map(|v| (v << 16) | (v * 101)).collect()));
    Set::new(&inputs)
}

/// Shape families the profiled generator cannot reach by chance, each chosen
/// because it drives a distinct path through the engine.
pub fn gen_family(st: &mut u64, n: usize) -> Spec {
    let pick = |st: &mut u64| (splitmix64(st) as usize) % n;
    match splitmix64(st) % 10 {
        // Interior AND over an *expanding* subtree, under a non-narrowing root:
        // the only shape where per-node hole-punch push-down would pay.
        0 => or(vec![
            and(vec![leaf(pick(st)), or(vec![leaf(pick(st)), leaf(pick(st))])]),
            leaf(pick(st)),
        ]),
        // Deep left-nested chain: exercises step-list splicing, which is
        // quadratic in depth, where the profiled generator builds balanced trees.
        1 => {
            let mut e = leaf(pick(st));
            for _ in 0..(8 + splitmix64(st) % 16) {
                e = and(vec![e, leaf(pick(st))]);
            }
            e
        }
        // Very high fan-in flat folds (16..48 operands).
        2 => {
            let w = 16 + (splitmix64(st) as usize) % 33;
            let kids = (0..w).map(|_| leaf(pick(st))).collect();
            if splitmix64(st).is_multiple_of(2) { or(kids) } else { and(kids) }
        }
        // Empty subtree feeding an expensive sibling: the short-circuit guard.
        3 => {
            let x = pick(st);
            and(vec![diff(leaf(x), leaf(x)), or((0..6).map(|_| leaf(pick(st))).collect())])
        }
        // Repeated identical operand (idempotent fold).
        4 => {
            let x = pick(st);
            let kids = (0..(2 + splitmix64(st) as usize % 5)).map(|_| leaf(x)).collect();
            if splitmix64(st).is_multiple_of(2) { and(kids) } else { or(kids) }
        }
        // Nested differences — DIFF never flattens, so each is its own fold.
        5 => {
            let mut e = leaf(pick(st));
            for _ in 0..(2 + splitmix64(st) % 4) {
                e = diff(e, leaf(pick(st)));
            }
            e
        }
        // Disjoint-key AND: the planner proves it empty before reading a byte.
        6 => and(vec![leaf(17 + (splitmix64(st) as usize) % 2), leaf(21 + (splitmix64(st) as usize) % 3)]),
        // Run LHS minus bitmap RHS at shared keys (the BUG-3 shape), nested so
        // it runs partner-major.
        7 => and(vec![diff(leaf(24), leaf(25)), leaf(pick(st))]),
        // Extreme skew: a tiny leaf intersected with the widest ones.
        8 => and(vec![leaf(26), leaf(pick(st)), leaf(pick(st))]),
        // A wide OR of DIFFs — every operand is a non-flattened subtree, so the
        // parent guards each one.
        _ => or((0..(3 + splitmix64(st) as usize % 4))
            .map(|_| diff(leaf(pick(st)), leaf(pick(st))))
            .collect()),
    }
}

/// A leaf whose blocks span keys `[k0, k1)`, `per_key` random values each — used
/// to build key-selective trees (a narrow band ∩ wide bands) for hole-punching.
pub fn band(k0: u16, k1: u16, per_key: u32, st: &mut u64) -> Vec<u32> {
    let mut v = Vec::new();
    for k in k0..k1 {
        for _ in 0..per_key {
            v.push(((k as u32) << 16) | (splitmix64(st) % 65536) as u32);
        }
    }
    sorted(v)
}

/// Named realistic shapes over the mixed pool (indices chosen for variety).
pub fn named() -> Vec<(&'static str, Spec)> {
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
pub struct Profile {
    leaf_pct: u64,
    diff_pct: u64,
    or_pct: u64,
    wide: bool,
}

/// One random tree under `budget` total leaves and `depth` remaining levels.
pub fn gen_profiled(st: &mut u64, budget: usize, depth: usize, prof: &Profile, n: usize) -> Spec {
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
    let max_w = if prof.wide && splitmix64(st).is_multiple_of(8) { budget } else { 2 + budget.min(6) };
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

/// Realized depth of a spec (a leaf counts 1).
pub fn depth_of(spec: &Spec) -> usize {
    match spec {
        Spec::Leaf(_) => 1,
        Spec::And(cs) | Spec::Or(cs) => 1 + cs.iter().map(depth_of).max().unwrap_or(0),
        Spec::Diff(a, b) => 1 + depth_of(a).max(depth_of(b)),
    }
}

/// A "pillar" tree hitting both corpus extremes at once: exactly `LEAVES`
/// leaves AND a realized depth of `DEPTH`. Built bottom-up along a spine of
/// nested nodes, hanging the remaining leaves across its levels at random
/// (random ops, random spine position per level).
pub fn gen_pillar(st: &mut u64, pool_len: usize) -> Spec {
    const LEAVES: usize = 100;
    const DEPTH: usize = 15;
    const SPINE: usize = DEPTH - 1; // internal levels above the deepest leaf

    // ≥1 extra leaf per spine level (every node needs ≥2 children); scatter
    // the rest randomly across the levels.
    let mut extra = [1usize; SPINE];
    for _ in 0..LEAVES - 1 - SPINE {
        extra[(splitmix64(st) as usize) % SPINE] += 1;
    }

    let rand_leaf = |st: &mut u64| Spec::Leaf((splitmix64(st) as usize) % pool_len);
    let mut cur = rand_leaf(st);
    for &m in &extra {
        let r = splitmix64(st) % 100;
        if r < 20 && m == 1 {
            // Diff level (binary): the spine randomly on either side.
            let l = rand_leaf(st);
            cur = if splitmix64(st).is_multiple_of(2) { diff(cur, l) } else { diff(l, cur) };
            continue;
        }
        let mut kids: Vec<Spec> = (0..m).map(|_| rand_leaf(st)).collect();
        kids.insert((splitmix64(st) as usize) % (kids.len() + 1), cur);
        cur = if r < 60 { and(kids) } else { or(kids) };
    }
    debug_assert_eq!(count_leaves(&cur), LEAVES);
    debug_assert_eq!(depth_of(&cur), DEPTH);
    cur
}

/// The 25k-tree corpus: up to 100 leaves and 15 levels per tree, deterministic.
/// The last 100 trees are pillars — guaranteed 100-leaf AND 15-deep.
pub fn corpus_specs(n_trees: usize, pool_len: usize) -> Vec<Spec> {
    let mut st = 0x00C0_2B05_2026_u64;
    let pillars = 100.min(n_trees);
    let mut specs: Vec<Spec> = (0..n_trees - pillars)
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
                wide: splitmix64(&mut st).is_multiple_of(3),
            };
            gen_profiled(&mut st, budget, depth, &prof, pool_len)
        })
        .collect();
    specs.extend((0..pillars).map(|_| gen_pillar(&mut st, pool_len)));
    // Same again from the shape families — structurally different trees, not
    // more samples of the same distribution.
    specs.extend((0..n_trees).map(|_| gen_family(&mut st, pool_len)));
    let full = specs.iter().filter(|s| count_leaves(s) == 100 && depth_of(s) == 15).count();
    assert!(full >= pillars, "corpus must include ≥{pillars} full-extreme trees");
    specs
}


/// Every leaf index in the tree, in visit order.
pub fn collect_leaves(s: &Spec, out: &mut Vec<usize>) {
    match s {
        Spec::Leaf(i) => out.push(*i),
        Spec::And(c) | Spec::Or(c) => c.iter().for_each(|x| collect_leaves(x, out)),
        Spec::Diff(a, b) => { collect_leaves(a, out); collect_leaves(b, out); }
    }
}
