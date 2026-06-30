//! Native run-container ops: differential vs roaring on dense run inputs.
//!
//! Inputs are a few long ranges per key (card ≫ 4096, run-encoded), so the
//! kernels take the native run path (Run ∩ / ∪ / − Run) rather than expanding
//! to a bitmap. Results are compared value-for-value against roaring.
#![cfg(feature = "roaring")]

use frostbit::{difference_fast, intersect_fast, union_fast, BitmapExpr, FrozenBitmap, FrozenBitmapView};
use roaring::RoaringBitmap;

fn splitmix64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn at(k: u16, lo: u32) -> u32 {
    ((k as u32) << 16) | lo
}

/// `keys` containers, each holding `nranges` long ranges of `rlen` (phase-
/// shifted) — dense and run-encoded.
fn run_set(keys: u16, nranges: u32, rlen: u32, phase: u32) -> Vec<u32> {
    let stride = 65536 / nranges;
    let mut v = Vec::new();
    for k in 0..keys {
        for i in 0..nranges {
            let start = (i * stride + phase) % 65536;
            let end = (start + rlen).min(65535);
            for lo in start..=end {
                v.push(at(k, lo));
            }
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

fn fz(v: &[u32]) -> FrozenBitmap {
    let mut b = frostbit::FrozenBitmapBuilder::new();
    b.extend_sorted(v.iter().copied());
    b.finish()
}
fn rb(v: &[u32]) -> RoaringBitmap {
    v.iter().copied().collect()
}
fn fv(b: &FrozenBitmap) -> Vec<u32> {
    b.view().iter().collect()
}

fn check(inputs: &[Vec<u32>]) {
    let fbs: Vec<FrozenBitmap> = inputs.iter().map(|v| fz(v)).collect();
    let views: Vec<FrozenBitmapView<'_>> = fbs.iter().map(|b| b.view()).collect();
    let rbs: Vec<RoaringBitmap> = inputs.iter().map(|v| rb(v)).collect();

    let r_and = rbs.iter().skip(1).fold(rbs[0].clone(), |a, b| &a & b);
    let r_or = rbs.iter().fold(RoaringBitmap::new(), |a, b| &a | b);
    let r_diff = rbs.iter().skip(1).fold(rbs[0].clone(), |a, b| &a - b);

    assert_eq!(fv(&intersect_fast(&views)), r_and.iter().collect::<Vec<_>>(), "AND");
    assert_eq!(fv(&union_fast(&views)), r_or.iter().collect::<Vec<_>>(), "OR");
    assert_eq!(fv(&difference_fast(&views)), r_diff.iter().collect::<Vec<_>>(), "DIFF");
}

#[test]
fn run_ops_match_roaring() {
    // 2-way through 5-way, varied range counts / lengths / phases.
    let mut st = 0x12345u64;
    for _ in 0..200 {
        let n = 2 + (splitmix64(&mut st) % 4) as usize;
        let keys = 1 + (splitmix64(&mut st) % 4) as u16;
        let nranges = 2 + (splitmix64(&mut st) % 6) as u32;
        let rlen = 2000 + (splitmix64(&mut st) % 9000) as u32;
        let inputs: Vec<Vec<u32>> = (0..n)
            .map(|_| run_set(keys, nranges, rlen, (splitmix64(&mut st) % 65536) as u32))
            .collect();
        check(&inputs);
    }
}

#[test]
fn run_minus_array_matches_roaring() {
    // Dense run minuend, sparse-array subtrahends → run-splitting path.
    let mut st = 0x9911u64;
    for _ in 0..150 {
        let keys = 1 + (splitmix64(&mut st) % 3) as u16;
        let run = run_set(keys, 2 + (splitmix64(&mut st) % 5) as u32, 4000 + (splitmix64(&mut st) % 8000) as u32, 0);
        // 1-2 sparse array subtrahends in the same keys.
        let n_sub = 1 + (splitmix64(&mut st) % 2) as usize;
        let mut inputs = vec![run];
        for _ in 0..n_sub {
            let cnt = 200 + (splitmix64(&mut st) % 2000) as usize;
            let mut s = std::collections::BTreeSet::new();
            for _ in 0..cnt {
                let k = (splitmix64(&mut st) % keys.max(1) as u64) as u16;
                s.insert(at(k, (splitmix64(&mut st) % 65536) as u32));
            }
            inputs.push(s.into_iter().collect());
        }
        check(&inputs);
    }
}

#[test]
fn run_tree_matches_roaring() {
    // Arena-chained tree eval over run leaves: (a ∪ b) ∩ (c \ d).
    let a = fz(&run_set(3, 3, 9000, 0));
    let b = fz(&run_set(3, 3, 9000, 12000));
    let c = fz(&run_set(3, 4, 12000, 4000));
    let d = fz(&run_set(3, 2, 6000, 20000));
    let (ra, rbm, rc, rd) = (
        rb(&run_set(3, 3, 9000, 0)),
        rb(&run_set(3, 3, 9000, 12000)),
        rb(&run_set(3, 4, 12000, 4000)),
        rb(&run_set(3, 2, 6000, 20000)),
    );

    let expr = BitmapExpr::and([
        BitmapExpr::or([BitmapExpr::leaf(a.view()), BitmapExpr::leaf(b.view())]),
        BitmapExpr::difference(BitmapExpr::leaf(c.view()), BitmapExpr::leaf(d.view())),
    ]);
    let want = &(&ra | &rbm) & &(&rc - &rd);
    assert_eq!(fv(&expr.materialize()), want.iter().collect::<Vec<_>>());
}
