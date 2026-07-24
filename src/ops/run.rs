//! Native operations on run containers — `&[Run]` in, `&[Run]` out.
//!
//! Run ∩ / ∪ / − Run stay in run form (O(runs), not O(bitmap)), so dense
//! run-encoded containers fold without ever expanding to a bitmap. Each run is
//! `start..=start + len` inclusive (cardinality `len + 1`); output runs are
//! ascending and non-overlapping. Callers size `out` for the worst case
//! (`a.len() + b.len()`), which the kernels bound to `MAX_RUNS`.

use crate::api::container::Run;

#[inline]
fn run(start: u32, end: u32) -> Run {
    Run { start: start as u16, len: (end - start) as u16 }
}

/// `a ∩ b`. Returns `(num_runs, cardinality)`.
pub fn intersect(a: &[Run], b: &[Run], out: &mut [Run]) -> (usize, u32) {
    let (mut ai, mut bi, mut nr, mut card) = (0, 0, 0, 0u32);
    while ai < a.len() && bi < b.len() {
        let (as_, ae) = (a[ai].start as u32, a[ai].end() as u32);
        let (bs, be) = (b[bi].start as u32, b[bi].end() as u32);
        let (lo, hi) = (as_.max(bs), ae.min(be));
        if lo <= hi {
            out[nr] = run(lo, hi);
            nr += 1;
            card += hi - lo + 1;
        }
        if ae <= be {
            ai += 1;
        } else {
            bi += 1;
        }
    }
    (nr, card)
}

/// `a ∪ b`, coalescing overlapping/adjacent runs. Returns `(num_runs, card)`.
pub fn union(a: &[Run], b: &[Run], out: &mut [Run]) -> (usize, u32) {
    if a.is_empty() {
        return copy(b, out);
    }
    if b.is_empty() {
        return copy(a, out);
    }
    let (mut ai, mut bi, mut nr, mut card) = (0, 0, 0, 0u32);
    // Seed the current run with whichever input starts earliest.
    let (mut cs, mut ce) = if a[0].start <= b[0].start {
        ai = 1;
        (a[0].start as u32, a[0].end() as u32)
    } else {
        bi = 1;
        (b[0].start as u32, b[0].end() as u32)
    };
    loop {
        let a_s = a.get(ai).map_or(u32::MAX, |r| r.start as u32);
        let b_s = b.get(bi).map_or(u32::MAX, |r| r.start as u32);
        if a_s == u32::MAX && b_s == u32::MAX {
            break;
        }
        let (ns, ne) = if a_s <= b_s {
            ai += 1;
            (a_s, a[ai - 1].end() as u32)
        } else {
            bi += 1;
            (b_s, b[bi - 1].end() as u32)
        };
        if ns <= ce + 1 {
            ce = ce.max(ne);
        } else {
            out[nr] = run(cs, ce);
            card += ce - cs + 1;
            nr += 1;
            (cs, ce) = (ns, ne);
        }
    }
    out[nr] = run(cs, ce);
    card += ce - cs + 1;
    (nr + 1, card)
}

/// `a \ b`: for each `a` run, subtract overlapping `b` runs, emitting the
/// surviving fragments. Returns `(num_runs, cardinality)`.
pub fn diff(a: &[Run], b: &[Run], out: &mut [Run]) -> (usize, u32) {
    if b.is_empty() {
        return copy(a, out);
    }
    let (mut bi, mut nr, mut card) = (0, 0, 0u32);
    for ar in a {
        let (mut lo, hi) = (ar.start as u32, ar.end() as u32);
        // Skip b runs ending before this a run.
        while bi < b.len() && (b[bi].end() as u32) < lo {
            bi += 1;
        }
        let mut bj = bi;
        while bj < b.len() && lo <= hi {
            let (bs, be) = (b[bj].start as u32, b[bj].end() as u32);
            if bs > hi {
                break;
            }
            if bs > lo {
                out[nr] = run(lo, bs - 1);
                card += bs - lo;
                nr += 1;
            }
            lo = be + 1;
            bj += 1;
        }
        if lo <= hi {
            out[nr] = run(lo, hi);
            card += hi - lo + 1;
            nr += 1;
        }
    }
    (nr, card)
}

/// `a \ b` where `b` is a sorted point set (array container): split each run
/// around the points it contains. Returns `(num_runs, cardinality)`. Output is
/// bounded by `a.len() + b.len()` runs.
pub fn diff_array(a: &[Run], b: &[u16], out: &mut [Run]) -> (usize, u32) {
    if b.is_empty() {
        return copy(a, out);
    }
    let (mut bi, mut nr, mut card) = (0, 0, 0u32);
    for ar in a {
        let (mut lo, hi) = (ar.start as u32, ar.end() as u32);
        while bi < b.len() && (b[bi] as u32) < lo {
            bi += 1;
        }
        let mut bj = bi;
        while bj < b.len() && lo <= hi {
            let v = b[bj] as u32;
            if v > hi {
                break;
            }
            if v > lo {
                out[nr] = run(lo, v - 1);
                card += v - lo;
                nr += 1;
            }
            lo = v + 1;
            bj += 1;
        }
        if lo <= hi {
            out[nr] = run(lo, hi);
            card += hi - lo + 1;
            nr += 1;
        }
    }
    (nr, card)
}

#[inline]
fn copy(src: &[Run], out: &mut [Run]) -> (usize, u32) {
    out[..src.len()].copy_from_slice(src);
    (src.len(), src.iter().map(|r| r.len as u32 + 1).sum())
}
