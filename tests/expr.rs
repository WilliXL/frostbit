//! Expression-tree evaluation: differential vs a roaring oracle on random trees.
#![cfg(feature = "roaring")]

use std::collections::BTreeSet;

use frostbit::{BitmapExpr, FrozenBitmap};
use roaring::RoaringBitmap;

mod support;
use support::splitmix64;


fn fz(values: &[u32]) -> FrozenBitmap {
    let mut b = frostbit::FrozenBitmapBuilder::new();
    b.extend_sorted(values.iter().copied());
    b.finish()
}

/// A pool of leaf bitmaps and the matching roaring oracles.
struct Pool {
    frozen: Vec<FrozenBitmap>,
    roaring: Vec<RoaringBitmap>,
}

impl Pool {
    fn new(st: &mut u64, n: usize) -> Self {
        let mut frozen = Vec::new();
        let mut roaring = Vec::new();
        for _ in 0..n {
            let cnt = (splitmix64(st) % 4000) as usize;
            let spread = 1u64 << (16 + (splitmix64(st) % 10));
            let mut set = BTreeSet::new();
            for _ in 0..cnt {
                set.insert((splitmix64(st) % spread) as u32);
            }
            let vals: Vec<u32> = set.iter().copied().collect();
            frozen.push(fz(&vals));
            roaring.push(vals.into_iter().collect());
        }
        Pool { frozen, roaring }
    }
}

/// Build matched (frostbit expr, roaring result) for a random tree of `depth`.
fn random_tree<'a>(
    st: &mut u64,
    pool: &'a Pool,
    depth: u32,
) -> (BitmapExpr<'a>, RoaringBitmap) {
    if depth == 0 || splitmix64(st).is_multiple_of(3) {
        let i = (splitmix64(st) as usize) % pool.frozen.len();
        return (BitmapExpr::leaf(pool.frozen[i].view()), pool.roaring[i].clone());
    }
    match splitmix64(st) % 3 {
        0 => {
            let n = 2 + (splitmix64(st) % 3) as usize;
            let mut exprs = Vec::new();
            let mut acc: Option<RoaringBitmap> = None;
            for _ in 0..n {
                let (e, r) = random_tree(st, pool, depth - 1);
                exprs.push(e);
                acc = Some(match acc {
                    None => r,
                    Some(a) => a & r,
                });
            }
            (BitmapExpr::and(exprs), acc.unwrap())
        }
        1 => {
            let n = 2 + (splitmix64(st) % 3) as usize;
            let mut exprs = Vec::new();
            let mut acc = RoaringBitmap::new();
            for _ in 0..n {
                let (e, r) = random_tree(st, pool, depth - 1);
                exprs.push(e);
                acc |= r;
            }
            (BitmapExpr::or(exprs), acc)
        }
        _ => {
            let (le, lr) = random_tree(st, pool, depth - 1);
            let (re, rr) = random_tree(st, pool, depth - 1);
            (BitmapExpr::difference(le, re), lr - rr)
        }
    }
}

fn assert_tree(expr: &BitmapExpr<'_>, want: &RoaringBitmap) {
    let got: Vec<u32> = expr.materialize().view().iter().collect();
    assert_eq!(got, want.iter().collect::<Vec<_>>());
}

#[test]
fn hand_built_trees() {
    let a = fz(&[1, 2, 3, 4, 5, 6]);
    let b = fz(&[2, 4, 6, 8]);
    let c = fz(&[4, 5, 6, 7]);
    // (a ∪ b) ∩ c  =  {4,5,6}
    let e = BitmapExpr::and([
        BitmapExpr::or([BitmapExpr::leaf(a.view()), BitmapExpr::leaf(b.view())]),
        BitmapExpr::leaf(c.view()),
    ]);
    assert_eq!(e.materialize().view().iter().collect::<Vec<_>>(), vec![4, 5, 6]);

    // a \ (b ∩ c)  =  a \ {4,6}  =  {1,2,3,5}
    let e = BitmapExpr::difference(
        BitmapExpr::leaf(a.view()),
        BitmapExpr::and([BitmapExpr::leaf(b.view()), BitmapExpr::leaf(c.view())]),
    );
    assert_eq!(e.materialize().view().iter().collect::<Vec<_>>(), vec![1, 2, 3, 5]);
}

#[test]
fn random_trees_match_roaring() {
    let mut st = 0x7BEE_2026_u64;
    let pool = Pool::new(&mut st, 12);
    for _ in 0..400 {
        let depth = 1 + (splitmix64(&mut st) % 5) as u32;
        let (expr, want) = random_tree(&mut st, &pool, depth);
        assert_tree(&expr, &want);
    }
}

/// Short-circuit is result-preserving. An AND (or DIFF) with an empty
/// non-flattened subtree operand must yield exactly the oracle — empty for the
/// AND, and the lhs-driven result for the DIFF — whether or not the guard fires.
#[test]
fn short_circuit_empty_subtree() {
    let x = fz(&[1, 2, 3]);
    let big = fz(&(0..5000).collect::<Vec<_>>());
    // diff(x, x) = ∅ (a non-flattened subtree); AND(∅, big) = ∅.
    let and = BitmapExpr::and([
        BitmapExpr::difference(BitmapExpr::leaf(x.view()), BitmapExpr::leaf(x.view())),
        BitmapExpr::leaf(big.view()),
    ]);
    assert!(and.materialize().view().iter().next().is_none());

    // DIFF(∅, big) = ∅ — the lhs guard fires.
    let d = BitmapExpr::difference(
        BitmapExpr::difference(BitmapExpr::leaf(x.view()), BitmapExpr::leaf(x.view())),
        BitmapExpr::leaf(big.view()),
    );
    assert!(d.materialize().view().iter().next().is_none());

    // A non-empty subtree must NOT short-circuit: AND(OR(x, big), big) = big ∩ … = x∪big ∩ big = big.
    let keep = BitmapExpr::and([
        BitmapExpr::or([BitmapExpr::leaf(x.view()), BitmapExpr::leaf(big.view())]),
        BitmapExpr::leaf(big.view()),
    ]);
    let got: Vec<u32> = keep.materialize().view().iter().collect();
    assert_eq!(got, (0..5000).collect::<Vec<_>>());
}

/// Auto hole-punching is result-preserving: an AND-rooted tree (the only shape
/// the analyzer derives a mask for) must yield exactly the roaring oracle, with
/// random — often OR/DIFF — branches underneath so the mask prunes dead keys
/// inside nested subtrees.
#[test]
fn and_root_trees_match_roaring() {
    let mut st = 0x50FF_2026_u64;
    let pool = Pool::new(&mut st, 12);
    for _ in 0..400 {
        let n = 2 + (splitmix64(&mut st) % 4) as usize;
        let depth = 1 + (splitmix64(&mut st) % 4) as u32;
        let mut exprs = Vec::new();
        let mut acc: Option<RoaringBitmap> = None;
        for _ in 0..n {
            let (e, r) = random_tree(&mut st, &pool, depth);
            exprs.push(e);
            acc = Some(match acc {
                None => r,
                Some(a) => a & r,
            });
        }
        let expr = BitmapExpr::and(exprs);
        assert_tree(&expr, &acc.unwrap());
    }
}

/// Flattening must reach the *analyzer*, not just the step list.
///
/// A nested same-op tree and its flat spelling are the same operation, so they
/// must plan identically — same operand count, same per-key accumulator forms,
/// hence the same cost. When `splice` inlined a flattened child's steps but
/// handed `combine` the child's merged shape, the executor folded k operands
/// while the analyzer modelled one pre-unioned container carrying k operands'
/// fan-in; that reads far denser than the truth and promoted array keys to
/// bitmap accumulators, making nested ORs up to 10x their flat equivalent.
///
/// Results were always correct, so only timing could catch it. This asserts the
/// structural property instead: identical results *and* an identical byte image,
/// at any nesting depth, since the serialized container forms are what diverged.
#[test]
fn nested_same_op_plans_like_its_flat_spelling() {
    let mut st = 0x11A7_2026_u64;
    let pool = Pool::new(&mut st, 10);
    let l = |i: usize| BitmapExpr::leaf(pool.frozen[i].view());

    for (flat, nested) in [
        (
            BitmapExpr::or([l(0), l(1), l(2), l(3), l(4)]),
            BitmapExpr::or([l(0), BitmapExpr::or([l(1), l(2), l(3), l(4)])]),
        ),
        (
            // Right-deep: flattening has to be transitive, not one level.
            BitmapExpr::or([l(5), l(6), l(7), l(8), l(9)]),
            BitmapExpr::or([
                l(5),
                BitmapExpr::or([l(6), BitmapExpr::or([l(7), BitmapExpr::or([l(8), l(9)])])]),
            ]),
        ),
        (
            BitmapExpr::and([l(0), l(1), l(2), l(3)]),
            BitmapExpr::and([BitmapExpr::and([l(0), l(1)]), BitmapExpr::and([l(2), l(3)])]),
        ),
    ] {
        let (a, b) = (flat.materialize(), nested.materialize());
        assert_eq!(a.view().iter().collect::<Vec<_>>(), b.view().iter().collect::<Vec<_>>());
        assert_eq!(a.as_bytes(), b.as_bytes(), "nested picked different container forms");
    }

    // A DIFF between them must stay unflattened (not associative) yet still
    // agree with the oracle, so the splice change cannot have leaked into it.
    let d = BitmapExpr::difference(
        BitmapExpr::or([l(0), BitmapExpr::or([l(1), l(2)])]),
        BitmapExpr::or([l(3), l(4)]),
    );
    let want = (&(&pool.roaring[0] | &pool.roaring[1]) | &pool.roaring[2])
        - (&pool.roaring[3] | &pool.roaring[4]);
    assert_eq!(d.materialize().view().iter().collect::<Vec<_>>(), want.iter().collect::<Vec<_>>());
}
