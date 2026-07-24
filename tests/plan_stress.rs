//! Exhaustive stress for the analysis pass: prove every slot capacity is a
//! sufficient upper bound so execution never allocates. Ground truth is
//! independent — each op's result is computed with `roaring`, re-encoded with
//! the builder, and its actual per-container bytes compared against the plan.
//!
//! Failure mode under test is niche (the monorepo saw ~1 in 10k), so this
//! sweeps boundary cardinalities, max-run shapes, container-key edges, value
//! extremes, and mixed encodings across thousands of randomized cases.
#![cfg(feature = "internals")]

use std::collections::{BTreeMap, BTreeSet};

use frostbit::format::*;
use frostbit::ops::cursor::ContainerCursor;
use frostbit::ops::analyze::plan::{fast_container_bytes, plan_diff, plan_intersect, plan_union, Op, Plan};
use frostbit::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};
use roaring::RoaringBitmap;

// --- builders / oracle ------------------------------------------------------

fn build(values: &[u32], standard: bool) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    if standard {
        b.finish_standard()
    } else {
        b.finish()
    }
}

fn rb(values: &[u32]) -> RoaringBitmap {
    RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap()
}

fn keys_of(values: &[u32]) -> BTreeSet<u16> {
    values.iter().map(|&v| (v >> 16) as u16).collect()
}

/// Per-key `(cardinality, stored container bytes)` of a value set, as the
/// builder would encode it — the minimal valid encoding (independent of plan).
fn result_meta(values: &[u32]) -> BTreeMap<u16, (u32, usize)> {
    let mut m = BTreeMap::new();
    if values.is_empty() {
        return m;
    }
    let bm = build(values, true);
    let v = bm.view();
    let mut c = ContainerCursor::new(&v);
    while c.peek_key().is_some() {
        let cr = c.get();
        m.insert(cr.key, (cr.card, cr.stored_bytes()));
        c.advance();
    }
    m
}

fn plan_keys(p: &Plan) -> Vec<u16> {
    p.slots.iter().map(|s| s.key).collect()
}

fn caps(p: &Plan) -> BTreeMap<u16, u32> {
    p.slots.iter().map(|s| (s.key, s.capacity)).collect()
}

/// Assert a plan's slots are sufficient for `result`, given the op and the raw
/// input value sets (for structural keys and DIFF rhs membership).
fn assert_sufficient(p: &Plan, op: Op, inputs: &[Vec<u32>], result: &[u32], label: &str) {
    let keys = plan_keys(p);
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "{label}: plan keys not ascending/distinct");

    // Structural key set.
    let structural: BTreeSet<u16> = match op {
        Op::Intersect => inputs
            .iter()
            .map(|v| keys_of(v))
            .reduce(|a, b| a.intersection(&b).copied().collect())
            .unwrap_or_default(),
        Op::Union => inputs.iter().flat_map(|v| keys_of(v)).collect(),
        Op::Diff => keys_of(&inputs[0]),
    };
    assert_eq!(keys.iter().copied().collect::<BTreeSet<_>>(), structural, "{label}: structural keys");

    let cap = caps(p);
    for s in &p.slots {
        assert!(s.capacity as usize <= BITMAP_BYTES, "{label}: cap>{BITMAP_BYTES} at {}", s.key);
    }

    // RHS key union for DIFF verbatim detection.
    let rhs_keys: BTreeSet<u16> = if op == Op::Diff {
        inputs[1..].iter().flat_map(|v| keys_of(v)).collect()
    } else {
        BTreeSet::new()
    };

    for (k, (card, stored)) in result_meta(result) {
        let c = *cap.get(&k).unwrap_or_else(|| panic!("{label}: result key {k} missing from plan"));
        // Universal: slot must hold the minimally-encoded result container.
        assert!(
            c as usize >= stored,
            "{label}: key {k} cap {c} < compacted result {stored} (card {card})"
        );
        // Execution-form: AND/OR emit array-or-bitmap; DIFF too where it shrinks
        // (a key present in some rhs). DIFF verbatim keys keep the input form,
        // already covered by the universal check above.
        let emits_fast = match op {
            Op::Intersect | Op::Union => true,
            Op::Diff => rhs_keys.contains(&k),
        };
        if emits_fast {
            assert!(
                c as usize >= fast_container_bytes(card),
                "{label}: key {k} cap {c} < fast {} (card {card})",
                fast_container_bytes(card)
            );
        }
    }
}

fn check_case(inputs: &[Vec<u32>], builds: &[bool]) {
    let bms: Vec<FrozenBitmap> =
        inputs.iter().enumerate().map(|(i, v)| build(v, builds[i % builds.len()])).collect();
    let views: Vec<FrozenBitmapView<'_>> = bms.iter().map(|b| b.view()).collect();

    let mut inter = rb(&inputs[0]);
    let mut uni = rb(&inputs[0]);
    let mut diff = rb(&inputs[0]);
    for v in &inputs[1..] {
        inter &= rb(v);
        uni |= rb(v);
        diff -= rb(v);
    }
    let inter: Vec<u32> = inter.iter().collect();
    let uni: Vec<u32> = uni.iter().collect();
    let diff: Vec<u32> = diff.iter().collect();

    assert_sufficient(&plan_intersect(&views), Op::Intersect, inputs, &inter, "AND");
    assert_sufficient(&plan_union(&views), Op::Union, inputs, &uni, "OR");
    assert_sufficient(&plan_diff(&views), Op::Diff, inputs, &diff, "DIFF");
}

// --- generators -------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn at(key: u16, lo: u16) -> u32 {
    ((key as u32) << 16) | lo as u32
}

/// `n` consecutive lows in one key → a single run.
fn run_block(key: u16, n: u32) -> Vec<u32> {
    (0..n).map(|i| at(key, i as u16)).collect()
}

/// `runs` runs of length `len` (≥3 ⇒ run encoding wins) with unit gaps in one key.
fn many_runs(key: u16, runs: usize, len: u16) -> Vec<u32> {
    let mut out = Vec::new();
    let mut lo = 0u32;
    for _ in 0..runs {
        for j in 0..len as u32 {
            out.push(at(key, (lo + j) as u16));
        }
        lo += len as u32 + 1;
        if lo > 0xFFFF {
            break;
        }
    }
    out
}

/// Sparse arrays / dense bitmaps / single values, by profile.
fn gen_input(st: &mut u64) -> Vec<u32> {
    let profile = splitmix64(st) % 8;
    let mut s: BTreeSet<u32> = BTreeSet::new();
    match profile {
        0 => {
            // sparse over a wide key range → arrays
            let n = (splitmix64(st) % 3000) as usize;
            for _ in 0..n {
                s.insert((splitmix64(st) % (1 << 22)) as u32);
            }
        }
        1 => {
            // a few consecutive run blocks
            for _ in 0..(1 + splitmix64(st) % 4) {
                let k = (splitmix64(st) % 8) as u16;
                let n = 1 + (splitmix64(st) % 5000) as u32;
                s.extend(run_block(k, n.min(0x1_0000)));
            }
        }
        2 => {
            // dense scattered in few keys → bitmaps
            for _ in 0..(1 + splitmix64(st) % 3) {
                let k = (splitmix64(st) % 4) as u16;
                let step = 1 + (splitmix64(st) % 3) as u16; // 1..3
                let n = 4000 + (splitmix64(st) % 12000) as u32;
                for i in 0..n {
                    let lo = i as u64 * step as u64;
                    if lo > 0xFFFF {
                        break;
                    }
                    s.insert(at(k, lo as u16));
                }
            }
        }
        3 => {
            // many runs near the run/bitmap boundary
            let k = (splitmix64(st) % 4) as u16;
            let runs = 1900 + (splitmix64(st) % 300) as usize; // straddles MAX_RUNS=2047
            s.extend(many_runs(k, runs, 3));
        }
        4 => {
            // boundary cardinalities in one key
            let k = (splitmix64(st) % 4) as u16;
            let card = *[1u32, 255, 256, 257, 4095, 4096, 4097, 8192, 65535, 65536]
                .get((splitmix64(st) % 10) as usize)
                .unwrap();
            // spread so it's an array/bitmap (step 1 would be a run)
            for i in 0..card {
                let lo = i as u64;
                if lo > 0xFFFF {
                    break;
                }
                s.insert(at(k, lo as u16));
            }
            if card <= 4096 {
                // make it sparse (array, not run): every other
                s.clear();
                for i in 0..card {
                    let lo = i as u64 * 2;
                    if lo > 0xFFFF {
                        break;
                    }
                    s.insert(at(k, lo as u16));
                }
            }
        }
        5 => {
            // container-boundary values
            for kk in 0..(1 + splitmix64(st) % 6) {
                let k = kk as u16;
                s.insert(at(k, 0));
                s.insert(at(k, 0xFFFF));
                s.insert(at(k, 0xFFFE));
            }
        }
        6 => {
            // single values across many keys → inline when compact
            let n = (splitmix64(st) % 130) as u16;
            for i in 0..n {
                s.insert(at(i, (splitmix64(st) % 0x1_0000) as u16));
            }
        }
        _ => {
            // extremes
            s.insert(0);
            s.insert(u32::MAX);
            s.insert(u32::MAX - 1);
            s.insert(0xFFFF);
            s.insert(0x1_0000);
        }
    }
    s.into_iter().collect()
}

// --- tests ------------------------------------------------------------------

#[test]
fn deterministic_boundary_sweep() {
    let cards = [0u32, 1, 2, 255, 256, 257, 4095, 4096, 4097, 8191, 8192, 65535, 65536];
    // single-key arrays at every boundary card, pairwise across all ops.
    for &ca in &cards {
        for &cb in &cards {
            let a: Vec<u32> = (0..ca).map(|i| at(0, (i % 0x1_0000) as u16)).collect();
            let b: Vec<u32> = (0..cb).map(|i| at(0, (i % 0x1_0000) as u16)).collect();
            // de-dup wrap for the 65536 case
            let a: Vec<u32> = a.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
            let b: Vec<u32> = b.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
            if a.is_empty() && b.is_empty() {
                continue;
            }
            let a = if a.is_empty() { vec![at(0, 0)] } else { a };
            let b = if b.is_empty() { vec![at(1, 0)] } else { b };
            check_case(&[a, b], &[true, false]);
        }
    }
}

#[test]
fn deterministic_shape_matrix() {
    let shapes: Vec<Vec<u32>> = vec![
        run_block(0, 1000),
        run_block(0, 65_536),                       // one full container (run)
        (0..65_536).map(|i| at(0, i as u16)).collect(), // full container, dense
        many_runs(0, 2047, 3),                      // exactly MAX_RUNS
        many_runs(0, 2048, 3),                      // over MAX_RUNS
        (0..4096).map(|i| at(0, (i * 2) as u16)).collect(), // 4096-array boundary
        (0..4097).map(|i| at(0, (i + i / 4) as u16)).collect(),
        vec![at(0, 0), at(0, 0xFFFF)],
        vec![at(0xFFFF, 0), at(0xFFFF, 0xFFFF)],
        (0..130).map(|k| at(k, 7)).collect(),       // many single-value keys
        vec![0, u32::MAX],
    ];
    for a in &shapes {
        for b in &shapes {
            for &sa in &[true, false] {
                for &sb in &[true, false] {
                    check_case(&[a.clone(), b.clone()], &[sa, sb]);
                }
            }
        }
    }
}

#[test]
fn randomized_2way() {
    let mut st = 0x5EED_0001_u64;
    for _ in 0..5000 {
        let a = gen_input(&mut st);
        let b = gen_input(&mut st);
        if a.is_empty() && b.is_empty() {
            continue;
        }
        let a = if a.is_empty() { vec![1] } else { a };
        let b = if b.is_empty() { vec![2] } else { b };
        let s = [(splitmix64(&mut st) & 1) == 0, (splitmix64(&mut st) & 1) == 0];
        check_case(&[a, b], &s);
    }
}

#[test]
fn randomized_nway() {
    let mut st = 0xBEEF_0002_u64;
    for _ in 0..4000 {
        let n = 3 + (splitmix64(&mut st) % 4) as usize; // 3..6 inputs
        let mut inputs: Vec<Vec<u32>> = (0..n).map(|_| gen_input(&mut st)).collect();
        if inputs.iter().all(|v| v.is_empty()) {
            inputs[0] = vec![1];
        }
        for v in inputs.iter_mut() {
            if v.is_empty() {
                *v = vec![splitmix64(&mut st) as u32];
            }
        }
        let builds: Vec<bool> = (0..n).map(|_| (splitmix64(&mut st) & 1) == 0).collect();
        check_case(&inputs, &builds);
    }
}

#[test]
fn randomized_max_run_overlap() {
    // Stress total_runs near/over MAX_RUNS across OR (sum of input run counts).
    let mut st = 0xD15E_0003_u64;
    for _ in 0..2000 {
        let k = (splitmix64(&mut st) % 3) as u16;
        let ra = 800 + (splitmix64(&mut st) % 1400) as usize;
        let rb_ = 800 + (splitmix64(&mut st) % 1400) as usize;
        let a = many_runs(k, ra, 3);
        let b = many_runs(k, rb_, 3);
        check_case(&[a, b], &[true, true]);
    }
}
