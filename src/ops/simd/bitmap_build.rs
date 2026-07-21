//! Bitmap construction helpers: populate or clear bits from values and runs,
//! and extract set bits back to an array. These are scalar word operations that
//! autovectorize where it matters; they have no per-arch dispatch.

use super::Bitmap;
use crate::container::Run;

#[inline]
pub fn copy(dst: &mut Bitmap, src: &Bitmap) {
    dst.copy_from_slice(src);
}

#[inline]
pub fn clear(dst: &mut Bitmap) {
    dst.fill(0);
}

#[inline]
pub fn set_values(dst: &mut Bitmap, vals: &[u16]) {
    for &v in vals {
        dst[v as usize / 64] |= 1u64 << (v as usize % 64);
    }
}

#[inline]
pub fn clear_values(dst: &mut Bitmap, vals: &[u16]) {
    for &v in vals {
        dst[v as usize / 64] &= !(1u64 << (v as usize % 64));
    }
}

pub fn set_runs(dst: &mut Bitmap, runs: &[Run]) {
    for r in runs {
        fill(dst, r.start as usize, r.end() as usize, true);
    }
}

pub fn clear_runs(dst: &mut Bitmap, runs: &[Run]) {
    for r in runs {
        fill(dst, r.start as usize, r.end() as usize, false);
    }
}

/// Set (or clear) the inclusive bit range `[lo, hi]` via word masks.
#[inline]
fn fill(dst: &mut Bitmap, lo: usize, hi: usize, set: bool) {
    let (wl, wh) = (lo / 64, hi / 64);
    let lo_mask = u64::MAX << (lo % 64);
    let hi_mask = u64::MAX >> (63 - hi % 64);
    if wl == wh {
        let m = lo_mask & hi_mask;
        if set {
            dst[wl] |= m;
        } else {
            dst[wl] &= !m;
        }
    } else {
        let full = if set { u64::MAX } else { 0 };
        if set {
            dst[wl] |= lo_mask;
            dst[wh] |= hi_mask;
        } else {
            dst[wl] &= !lo_mask;
            dst[wh] &= !hi_mask;
        }
        dst[wl + 1..wh].fill(full);
    }
}
