//! Per-tree corpus audit: time every corpus tree on both engines and report
//! each tree where frostbit trails roaring by more than a threshold, with the
//! shape features needed to see what the offenders have in common.
//!
//! Not a criterion bench — a plain harness (`cargo bench --bench tree_audit`,
//! `+nightly --features roaring-simd` for the SIMD competitor). Timing is
//! min-of-N per tree, which discards scheduler noise.

use std::time::Instant;

#[path = "common.rs"]
mod common;
use common::*;
#[path = "treegen.rs"]
mod treegen;
use treegen::*;

const THRESHOLD_US: f64 = 3.0;
const REPS: usize = 5;

fn best_us(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    best
}

/// Shape features of one spec.
struct Feat {
    leaves: usize,
    depth: usize,
    ands: usize,
    ors: usize,
    diffs: usize,
    /// Leaves per pool class: [small array, medium array, dense, runs].
    class: [usize; 4],
    max_width: usize,
}

fn featurize(spec: &Spec, f: &mut Feat) {
    match spec {
        Spec::Leaf(i) => {
            let c = match i {
                0..=5 => 0,
                6..=9 => 1,
                10..=13 => 2,
                _ => 3,
            };
            f.class[c] += 1;
            f.leaves += 1;
        }
        Spec::And(cs) | Spec::Or(cs) => {
            if matches!(spec, Spec::And(_)) {
                f.ands += 1;
            } else {
                f.ors += 1;
            }
            f.max_width = f.max_width.max(cs.len());
            for c in cs {
                featurize(c, f);
            }
        }
        Spec::Diff(a, b) => {
            f.diffs += 1;
            featurize(a, f);
            featurize(b, f);
        }
    }
}

/// Compact structural dump: `&`=AND `|`=OR `\\`=DIFF, leaves by pool index.
fn dump(spec: &Spec, out: &mut String) {
    match spec {
        Spec::Leaf(i) => out.push_str(&i.to_string()),
        Spec::And(cs) | Spec::Or(cs) => {
            out.push(if matches!(spec, Spec::And(_)) { '&' } else { '|' });
            out.push('(');
            for (k, c) in cs.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                dump(c, out);
            }
            out.push(')');
        }
        Spec::Diff(a, b) => {
            out.push_str("\\(");
            dump(a, out);
            out.push(',');
            dump(b, out);
            out.push(')');
        }
    }
}

fn main() {
    let pool = mixed_pool();
    let specs = corpus_specs(25_000, pool.len());
    eprintln!("auditing {} trees vs {RB} (threshold {THRESHOLD_US} µs, min of {REPS})…", specs.len());

    let mut offenders: Vec<(usize, f64, f64)> = Vec::new();
    let (mut fb_total, mut rb_total) = (0f64, 0f64);
    for (i, spec) in specs.iter().enumerate() {
        let fb = best_us(|| {
            std::hint::black_box(build_fb(spec, &pool).materialize());
        });
        let rb = best_us(|| {
            std::hint::black_box(eval_rb(spec, &pool));
        });
        fb_total += fb;
        rb_total += rb;
        if fb > rb + THRESHOLD_US {
            offenders.push((i, fb, rb));
        }
    }

    offenders.sort_by(|a, b| (b.1 - b.2).total_cmp(&(a.1 - a.2)));
    println!(
        "\ncorpus totals: frostbit {:.1} ms, {RB} {:.1} ms  |  offenders (> {THRESHOLD_US} µs behind): {} of {}",
        fb_total / 1e3,
        rb_total / 1e3,
        offenders.len(),
        specs.len()
    );

    // Aggregate offender features vs the whole corpus.
    let agg = |idxs: &mut dyn Iterator<Item = usize>| -> (f64, f64, f64, f64, f64, [f64; 4], f64) {
        let (mut n, mut lv, mut dp, mut an, mut or_, mut df, mut cl, mut wd) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, [0f64; 4], 0f64);
        for i in idxs {
            let mut f = Feat { leaves: 0, depth: depth_of(&specs[i]), ands: 0, ors: 0, diffs: 0, class: [0; 4], max_width: 0 };
            featurize(&specs[i], &mut f);
            n += 1.0;
            lv += f.leaves as f64;
            dp += f.depth as f64;
            an += f.ands as f64;
            or_ += f.ors as f64;
            df += f.diffs as f64;
            for k in 0..4 {
                cl[k] += f.class[k] as f64;
            }
            wd = wd.max(f.max_width as f64);
        }
        let n = n.max(1.0);
        (lv / n, dp / n, an / n, or_ / n, df / n, cl.map(|x| x / n), wd)
    };
    let all = agg(&mut (0..specs.len()));
    let off = agg(&mut offenders.iter().map(|o| o.0));
    println!("            leaves  depth  ANDs  ORs  DIFFs  [sm  med  dense  runs]  maxw");
    println!(
        "corpus avg  {:6.1} {:6.1} {:5.1} {:4.1} {:6.1}  [{:.1} {:.1} {:.1} {:.1}]  {:.0}",
        all.0, all.1, all.2, all.3, all.4, all.5[0], all.5[1], all.5[2], all.5[3], all.6
    );
    println!(
        "offender avg{:6.1} {:6.1} {:5.1} {:4.1} {:6.1}  [{:.1} {:.1} {:.1} {:.1}]  {:.0}",
        off.0, off.1, off.2, off.3, off.4, off.5[0], off.5[1], off.5[2], off.5[3], off.6
    );

    println!("\nworst 5 structures:");
    for &(i, fb, rb) in offenders.iter().take(5) {
        let mut s = String::new();
        dump(&specs[i], &mut s);
        println!("  #{i} (fb {fb:.0} µs, rb {rb:.0} µs): {s}");
    }

    println!("\nworst 30:");
    println!("  idx    fb µs    rb µs   delta  leaves depth ANDs ORs DIFFs [sm med den run] maxw");
    for &(i, fb, rb) in offenders.iter().take(30) {
        let mut f = Feat { leaves: 0, depth: depth_of(&specs[i]), ands: 0, ors: 0, diffs: 0, class: [0; 4], max_width: 0 };
        featurize(&specs[i], &mut f);
        println!(
            "{i:6} {fb:8.1} {rb:8.1} {:7.1}  {:5} {:5} {:4} {:3} {:5} [{:3} {:3} {:3} {:3}] {:4}",
            fb - rb, f.leaves, f.depth, f.ands, f.ors, f.diffs,
            f.class[0], f.class[1], f.class[2], f.class[3], f.max_width
        );
    }
}
