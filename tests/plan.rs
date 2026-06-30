//! Analysis pass: container cursor + AND/OR/DIFF plans. Verifies the static
//! invariant that every output slot's capacity is a proven upper bound (so
//! execution never allocates), using a BTreeSet/roaring oracle.
#![cfg(feature = "internals")]

use std::collections::{BTreeMap, BTreeSet};

use frostbit::format::*;
use frostbit::ops::cursor::ContainerCursor;
use frostbit::ops::plan::{fast_container_bytes, plan_diff, plan_intersect, plan_union, Plan};
use frostbit::{FrozenBitmap, FrozenBitmapBuilder, FrozenBitmapView};

fn build_std(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish_standard()
}

fn build_compact(values: &[u32]) -> FrozenBitmap {
    let mut b = FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

fn keys_of(values: &[u32]) -> BTreeSet<u16> {
    values.iter().map(|&v| (v >> 16) as u16).collect()
}

fn result_cards(values: &[u32]) -> BTreeMap<u16, u32> {
    let mut m = BTreeMap::new();
    for &v in values {
        *m.entry((v >> 16) as u16).or_insert(0) += 1;
    }
    m
}

fn plan_keys(p: &Plan) -> Vec<u16> {
    p.slots.iter().map(|s| s.key).collect()
}

/// Shared invariants for any plan + its true result value set.
fn check_plan(p: &Plan, result: &[u32]) {
    let keys = plan_keys(p);
    // Ascending, distinct.
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "plan keys not ascending/distinct");
    // Every capacity is a valid container ceiling.
    for s in &p.slots {
        assert!(s.capacity >= 1, "zero capacity at key {}", s.key);
        assert!(s.capacity as usize <= BITMAP_BYTES, "capacity > bitmap at key {}", s.key);
    }
    let plan_set: BTreeSet<u16> = keys.iter().copied().collect();
    // Plan never drops a key that survives.
    for k in keys_of(result) {
        assert!(plan_set.contains(&k), "result key {k} missing from plan");
    }
    // Each surviving container fits its slot in op-ready form.
    let cap_at: BTreeMap<u16, u32> = p.slots.iter().map(|s| (s.key, s.capacity)).collect();
    for (k, card) in result_cards(result) {
        let cap = cap_at[&k];
        assert!(
            cap as usize >= fast_container_bytes(card),
            "key {k}: cap {cap} < fast bytes {} for card {card}",
            fast_container_bytes(card)
        );
    }
}

// --- cursor ----------------------------------------------------------------

#[test]
fn cursor_standard_metadata() {
    // run (key0), array (key2), bitmap (key3).
    let mut vals: Vec<u32> = (0..1000).collect();
    vals.extend((0..100).map(|i| 131_072 + i * 5));
    vals.extend((196_608..196_608 + 10_000).step_by(2));
    let bm = build_std(&vals);
    let v = bm.view();
    let mut c = ContainerCursor::new(&v);
    let mut seen = Vec::new();
    while let Some(k) = c.peek_key() {
        let cr = c.get();
        assert_eq!(cr.key, k);
        seen.push((cr.key, cr.typ, cr.card));
        c.advance();
    }
    assert_eq!(seen[0], (0, CT_RUN, 1000));
    assert_eq!(seen[1], (2, CT_ARRAY, 100));
    assert_eq!(seen[2], (3, CT_BITMAP, 5000));
}

#[test]
fn cursor_inline_groups_by_key() {
    let bm = build_compact(&[0, 1, 2, 65_536, 65_537, 131_072]);
    let v = bm.view();
    assert!(v.is_inline());
    let mut c = ContainerCursor::new(&v);
    let mut seen = Vec::new();
    while c.peek_key().is_some() {
        let cr = c.get();
        seen.push((cr.key, cr.typ, cr.card));
        c.advance();
    }
    assert_eq!(seen, vec![(0, CT_INLINE, 3), (1, CT_INLINE, 2), (2, CT_INLINE, 1)]);
}

#[test]
fn cursor_advance_to() {
    // keys 0, 1, 3 (no key 2).
    let bm = build_std(&[0, 65_536, 196_608]);
    let v = bm.view();
    let mut c = ContainerCursor::new(&v);
    assert!(c.advance_to(1)); // key 1 present
    assert!(!c.advance_to(2)); // no key 2 → stops at key 3
    assert_eq!(c.peek_key(), Some(3));
}

// --- shapes ----------------------------------------------------------------

/// (name, inputs). Exercises every container type, multi-key, disjoint/overlap.
fn op_cases() -> Vec<(&'static str, Vec<Vec<u32>>)> {
    let runa: Vec<u32> = (0..2000).collect();
    let arr: Vec<u32> = (0..300).map(|i| i * 7).collect();
    let bmp: Vec<u32> = (0..6000).map(|i| i * 2).collect();
    vec![
        ("disjoint", vec![vec![1, 2, 3], vec![10, 20, 30]]),
        ("overlap", vec![vec![1, 2, 3, 4, 5], vec![3, 4, 5, 6, 7]]),
        ("multikey", vec![vec![0, 65_536, 131_072], vec![1, 65_536, 196_608]]),
        ("run_vs_array", vec![runa.clone(), arr.clone()]),
        ("bitmap_vs_array", vec![bmp.clone(), arr.clone()]),
        ("run_vs_bitmap", vec![runa.clone(), bmp.clone()]),
        ("three_way", vec![arr.clone(), runa.clone(), bmp.clone()]),
        ("single", vec![arr.clone()]),
        ("one_empty", vec![arr.clone(), vec![]]),
    ]
}

fn rb(values: &[u32]) -> roaring::RoaringBitmap {
    roaring::RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap()
}

fn run_for_inputs(build: fn(&[u32]) -> FrozenBitmap) {
    for (name, inputs) in op_cases() {
        let bms: Vec<FrozenBitmap> = inputs.iter().map(|v| build(v)).collect();
        let views: Vec<FrozenBitmapView<'_>> = bms.iter().map(|b| b.view()).collect();

        // Intersect.
        let mut inter = rb(&inputs[0]);
        for v in &inputs[1..] {
            inter &= rb(v);
        }
        let inter_vals: Vec<u32> = inter.iter().collect();
        let p = plan_intersect(&views);
        check_plan(&p, &inter_vals);
        let structural: BTreeSet<u16> = inputs
            .iter()
            .map(|v| keys_of(v))
            .reduce(|a, b| a.intersection(&b).copied().collect())
            .unwrap();
        assert_eq!(plan_keys(&p), structural.into_iter().collect::<Vec<_>>(), "{name} AND keys");

        // Union.
        let mut uni = rb(&inputs[0]);
        for v in &inputs[1..] {
            uni |= rb(v);
        }
        let uni_vals: Vec<u32> = uni.iter().collect();
        let p = plan_union(&views);
        check_plan(&p, &uni_vals);
        let structural: BTreeSet<u16> = inputs.iter().flat_map(|v| keys_of(v)).collect();
        assert_eq!(plan_keys(&p), structural.into_iter().collect::<Vec<_>>(), "{name} OR keys");

        // Diff (A minus the rest).
        let mut diff = rb(&inputs[0]);
        for v in &inputs[1..] {
            diff -= rb(v);
        }
        let diff_vals: Vec<u32> = diff.iter().collect();
        let p = plan_diff(&views);
        check_plan(&p, &diff_vals);
        assert_eq!(
            plan_keys(&p),
            keys_of(&inputs[0]).into_iter().collect::<Vec<_>>(),
            "{name} DIFF keys"
        );
    }
}

#[test]
fn plans_over_standard_inputs() {
    run_for_inputs(build_std);
}

#[test]
fn plans_over_compact_inputs() {
    run_for_inputs(build_compact);
}

#[test]
fn empty_inputs() {
    assert!(plan_intersect(&[]).slots.is_empty());
    assert!(plan_union(&[]).slots.is_empty());
    assert!(plan_diff(&[]).slots.is_empty());
}

// --- scale + randomized ----------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
fn randomized_differential() {
    let mut st = 0xA11CE_u64;
    for _ in 0..200 {
        let mk = |st: &mut u64| -> Vec<u32> {
            let n = (splitmix64(st) % 4000) as usize;
            let mut s: BTreeSet<u32> = BTreeSet::new();
            let spread = 1u32 << (18 + (splitmix64(st) % 8) as u32); // varies block count
            for _ in 0..n {
                s.insert((splitmix64(st) % spread as u64) as u32);
            }
            s.into_iter().collect()
        };
        let a = mk(&mut st);
        let b = mk(&mut st);
        let c = mk(&mut st);
        let inputs = [a.as_slice(), b.as_slice(), c.as_slice()];
        let bms: Vec<FrozenBitmap> = inputs.iter().map(|v| build_std(v)).collect();
        let views: Vec<FrozenBitmapView<'_>> = bms.iter().map(|b| b.view()).collect();

        let inter: Vec<u32> = (rb(&a) & rb(&b) & rb(&c)).iter().collect();
        check_plan(&plan_intersect(&views), &inter);
        let uni: Vec<u32> = (rb(&a) | rb(&b) | rb(&c)).iter().collect();
        check_plan(&plan_union(&views), &uni);
        let diff: Vec<u32> = ((rb(&a) - rb(&b)) - rb(&c)).iter().collect();
        check_plan(&plan_diff(&views), &diff);
    }
}

#[test]
fn scale_multi_container() {
    // ~150 blocks each; intersection concentrated, union wide.
    let a: Vec<u32> = (0..10_000_000u32).step_by(11).collect();
    let b: Vec<u32> = (0..10_000_000u32).step_by(13).collect();
    let av = build_std(&a);
    let bv = build_std(&b);
    let views = [av.view(), bv.view()];

    let inter: Vec<u32> = (rb(&a) & rb(&b)).iter().collect();
    let p = plan_intersect(&views);
    assert!(p.num_slots() > 100);
    check_plan(&p, &inter);

    let uni: Vec<u32> = (rb(&a) | rb(&b)).iter().collect();
    check_plan(&plan_union(&views), &uni);

    let diff: Vec<u32> = (rb(&a) - rb(&b)).iter().collect();
    check_plan(&plan_diff(&views), &diff);
}
