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
            b.iter(|| black_box(frostbit::ops::analyze::plan::plan_diff(&fv)))
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
        bch.iter(|| black_box(frostbit::simd::array_diff(&a800, &b800, &mut out)))
    });
    k.bench_function("diff/660x800", |bch| {
        bch.iter(|| black_box(frostbit::simd::array_diff(&a660, &b800, &mut out)))
    });
    k.bench_function("intersect/800x800", |bch| {
        bch.iter(|| black_box(frostbit::simd::array_intersect(&a800, &b800, &mut out)))
    });
    k.bench_function("union/800x800", |bch| {
        bch.iter(|| black_box(frostbit::simd::array_union(&a800, &b800, &mut out)))
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
    g.bench_function("diff2/plan", |b| b.iter(|| black_box(frostbit::ops::analyze::plan::plan_diff(&rv2))));
    g.bench_function("diff2/into", |b| b.iter(|| black_box(frostbit::ops::kernels::diff_into(&rv2))));
    g.bench_function("diff2/full", |b| b.iter(|| black_box(difference_fast(&rv2))));
    g.bench_function(format!("diff2/{RB}"), |b| b.iter(|| black_box(rb_diff(&runs.rbs[..2]))));
    g.bench_function("and16/plan", |b| b.iter(|| black_box(frostbit::ops::analyze::plan::plan_intersect(&rv16))));
    g.bench_function("and16/into", |b| b.iter(|| black_box(frostbit::ops::kernels::intersect_into(&rv16))));
    g.bench_function("and16/full", |b| b.iter(|| black_box(intersect_fast(&rv16))));
    g.bench_function(format!("and16/{RB}"), |b| b.iter(|| black_box(rb_and(&runs.rbs[..16]))));
    g.finish();

    // Seed/order study shapes. skew: same 16 keys, cards alternating small/large
    // (per-key partner order matters); disjoint: same keys, non-overlapping
    // value bands (n-way AND is empty — range pre-test target).
    let mut st = 0x5EED_0DE2_u64;
    let skew: Vec<Vec<u32>> = (0..16)
        .map(|i| {
            let card = if i % 2 == 0 { 400 } else { 2500 };
            arrays(16, card, &mut st)
        })
        .collect();
    let skew = Set::new(&skew);
    let band = |lo: u32, hi: u32, per: u32, st: &mut u64| -> Vec<u32> {
        let mut v = Vec::new();
        for k in 0..16u16 {
            for _ in 0..per {
                v.push(((k as u32) << 16) | (lo + splitmix64(st) as u32 % (hi - lo)));
            }
        }
        sorted(v)
    };
    let disj: Vec<Vec<u32>> = (0..8)
        .map(|i| band(i * 8192, (i + 1) * 8192, 800, &mut st))
        .collect();
    let disj = Set::new(&disj);
    let mut g = c.benchmark_group("decomp_seed");
    for n in [8usize, 16] {
        let fv = skew.views(n);
        let rv = &skew.rbs[..n];
        g.bench_function(format!("skew/and{n}"), |b| b.iter(|| black_box(intersect_fast(&fv))));
        g.bench_function(format!("skew/and{n}/{RB}"), |b| b.iter(|| black_box(rb_and(rv))));
    }
    for n in [2usize, 8] {
        let fv = disj.views(n);
        let rv = &disj.rbs[..n];
        g.bench_function(format!("disjoint/and{n}"), |b| b.iter(|| black_box(intersect_fast(&fv))));
        g.bench_function(format!("disjoint/and{n}/{RB}"), |b| b.iter(|| black_box(rb_and(rv))));
    }
    // Non-degenerate wide run intersection: phase step 64 keeps all 16 windows
    // overlapping (the sweep's runs/16 cell annihilates — this one measures
    // merge throughput, not collapse).
    let mut ovl = Set::new(&(0..16).map(|i| run_ranges(16, 4, 6000, i * 64)).collect::<Vec<_>>());
    ovl.optimize_roaring();
    for n in [8usize, 16] {
        let fv = ovl.views(n);
        let rv = &ovl.rbs[..n];
        g.bench_function(format!("runs_ovl/and{n}"), |b| b.iter(|| black_box(intersect_fast(&fv))));
        g.bench_function(format!("runs_ovl/and{n}/{RB}"), |b| b.iter(|| black_box(rb_and(rv))));
    }
    g.finish();
}

// Memory/word-op roofline: unit throughput of every bitmap primitive at
// container scale, so cell times decompose against measured ceilings — plus
// the two bitmap→array extraction algorithms (current per-bit vs byte-table).
#[cfg(feature = "internals")]
fn membw(c: &mut Criterion) {
    use frostbit::container::{as_bitmap, as_bitmap_mut, Data};
    use frostbit::simd as k;

    let mut st = 0xBEEF_CAFE_u64;
    let mk_bitmap = |card: usize, st: &mut u64| -> Vec<u8> {
        let mut buf = vec![0u8; 8192];
        {
            let bm = as_bitmap_mut(&mut buf);
            let mut n = 0;
            while n < card {
                let v = (splitmix64(st) % 65536) as usize;
                if bm[v / 64] & (1 << (v % 64)) == 0 {
                    bm[v / 64] |= 1 << (v % 64);
                    n += 1;
                }
            }
        }
        buf
    };
    let a5000 = mk_bitmap(5000, &mut st);
    let b5000 = mk_bitmap(5000, &mut st);
    let a3900 = mk_bitmap(3900, &mut st);
    let a380 = mk_bitmap(380, &mut st);
    let vals800: Vec<u16> = {
        let mut s = std::collections::BTreeSet::new();
        while s.len() < 800 {
            s.insert((splitmix64(&mut st) % 65536) as u16);
        }
        s.into_iter().collect()
    };
    let mut dst = a5000.clone();
    let mut out16 = vec![0u16; 8192];

    // Byte-table extraction prototype: 256-row table of in-byte bit offsets.
    let table: Vec<[u16; 8]> = (0..256usize)
        .map(|b| {
            let (mut row, mut k) = ([0u16; 8], 0);
            for bit in 0..8 {
                if b & (1 << bit) != 0 {
                    row[k] = bit as u16;
                    k += 1;
                }
            }
            row
        })
        .collect();

    let mut g = c.benchmark_group("membw");
    g.bench_function("copy8k", |b| {
        b.iter(|| k::copy(as_bitmap_mut(&mut dst), black_box(as_bitmap(&a5000))))
    });
    g.bench_function("or8k", |b| {
        b.iter(|| k::or(as_bitmap_mut(&mut dst), black_box(as_bitmap(&b5000))))
    });
    g.bench_function("or_count8k", |b| {
        b.iter(|| black_box(k::or_count(as_bitmap_mut(&mut dst), black_box(as_bitmap(&b5000)))))
    });
    g.bench_function("and_count8k", |b| {
        b.iter(|| black_box(k::and_count(as_bitmap_mut(&mut dst), black_box(as_bitmap(&b5000)))))
    });
    g.bench_function("andnot_into_count8k", |b| {
        b.iter(|| {
            black_box(k::andnot_into_count(
                as_bitmap_mut(&mut dst),
                black_box(as_bitmap(&a5000)),
                black_box(as_bitmap(&b5000)),
            ))
        })
    });
    g.bench_function("popcount8k", |b| b.iter(|| black_box(k::popcount(as_bitmap(&a5000)))));
    g.bench_function("clear8k", |b| b.iter(|| k::clear(as_bitmap_mut(&mut dst))));
    g.bench_function("scatter800", |b| {
        b.iter(|| k::set_values(as_bitmap_mut(&mut dst), black_box(&vals800)))
    });
    // andnot_into_count variants: isolate the popcount accumulator chain.
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let (av, bv) = (a5000.clone(), b5000.clone());
        g.bench_function("andnot_into_4acc", |bch| {
            bch.iter(|| unsafe {
                let d = as_bitmap_mut(&mut dst).as_mut_ptr();
                let a = as_bitmap(&av).as_ptr();
                let b = as_bitmap(&bv).as_ptr();
                let (mut c0, mut c1, mut c2, mut c3) =
                    (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
                let mut i = 0usize;
                while i < 1024 {
                    let r0 = vbicq_u64(vld1q_u64(a.add(i)), vld1q_u64(b.add(i)));
                    let r1 = vbicq_u64(vld1q_u64(a.add(i + 2)), vld1q_u64(b.add(i + 2)));
                    let r2 = vbicq_u64(vld1q_u64(a.add(i + 4)), vld1q_u64(b.add(i + 4)));
                    let r3 = vbicq_u64(vld1q_u64(a.add(i + 6)), vld1q_u64(b.add(i + 6)));
                    vst1q_u64(d.add(i), r0);
                    vst1q_u64(d.add(i + 2), r1);
                    vst1q_u64(d.add(i + 4), r2);
                    vst1q_u64(d.add(i + 6), r3);
                    c0 = vpadalq_u8(c0, vcntq_u8(vreinterpretq_u8_u64(r0)));
                    c1 = vpadalq_u8(c1, vcntq_u8(vreinterpretq_u8_u64(r1)));
                    c2 = vpadalq_u8(c2, vcntq_u8(vreinterpretq_u8_u64(r2)));
                    c3 = vpadalq_u8(c3, vcntq_u8(vreinterpretq_u8_u64(r3)));
                    i += 8;
                }
                let s = vaddq_u16(vaddq_u16(c0, c1), vaddq_u16(c2, c3));
                black_box(vaddlvq_u16(s))
            })
        });
        g.bench_function("andnot_into_nocount", |bch| {
            bch.iter(|| unsafe {
                let d = as_bitmap_mut(&mut dst).as_mut_ptr();
                let a = as_bitmap(&av).as_ptr();
                let b = as_bitmap(&bv).as_ptr();
                let mut i = 0usize;
                while i < 1024 {
                    vst1q_u64(d.add(i), vbicq_u64(vld1q_u64(a.add(i)), vld1q_u64(b.add(i))));
                    i += 2;
                }
                black_box(d)
            })
        });
    }

    for (name, bmb, card) in [("extract3900", &a3900, 3900usize), ("extract380", &a380, 380)] {
        let data = Data::Bitmap(as_bitmap(bmb));
        g.bench_function(format!("{name}/current"), |b| {
            b.iter(|| black_box(data.write_sorted(black_box(&mut out16))))
        });
        g.bench_function(format!("{name}/byte_table"), |b| {
            b.iter(|| {
                let bm = as_bitmap(bmb);
                let mut n = 0usize;
                for (w, &word) in bm.iter().enumerate() {
                    if word == 0 {
                        continue;
                    }
                    let base = (w * 64) as u16;
                    for byte in 0..8 {
                        let bits = ((word >> (byte * 8)) & 0xFF) as usize;
                        if bits == 0 {
                            continue;
                        }
                        let row = &table[bits];
                        let off = base + (byte * 8) as u16;
                        for j in 0..8 {
                            out16[n + j] = row[j] + off;
                        }
                        n += bits.count_ones() as usize;
                    }
                }
                black_box(n)
            })
        });
        black_box(card);
    }
    g.finish();
}
#[cfg(not(feature = "internals"))]
fn membw(_c: &mut Criterion) {}
#[cfg(not(feature = "internals"))]
fn decomp(_c: &mut Criterion) {}

// TEMP (profiling): branch-predictability sweep. Same element counts and block
// counts per kernel; only the *pattern* differs — `alt` interleaves the sides
// (advance branch strictly alternates, fully predictable), `rand` scatters
// side membership (advance ~50/50 data-dependent). Overlap sweeps vary the
// intersect/diff hit rate (emit-loop density). Deltas isolate branch stalls.
#[cfg(feature = "internals")]
fn branches(c: &mut Criterion) {
    use frostbit::simd as k;
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
    sweep(c, "intersect", intersect_fast, rb_and, &sets);
    sweep(c, "union", union_fast, rb_or, &sets);
    sweep(c, "diff", difference_fast, rb_diff, &sets);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1200))
        .measurement_time(Duration::from_secs(4));
    targets = bench, decomp, branches, membw
}
criterion_main!(benches);
