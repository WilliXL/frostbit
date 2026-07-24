//! The one place per-key container decisions are made.
//!
//! Two passes size the same output: the cursor-driven flat planner
//! ([`plan`](super::plan), which sees live containers) and the shape-driven tree
//! analyzer ([`shape`](super::shape), which sees a child's summarized output).
//! They must reach **identical** conclusions — a slot capacity is only "proven"
//! if it matches what the kernel will actually build there, so a rule that
//! drifts between the two silently under-sizes an arena.
//!
//! Keeping the rules here means neither pass can restate one. Each is a small
//! `#[inline]` function over `Copy` stats, so routing through it costs nothing.

use crate::format::*;
use crate::ops::plan::{union_promotes, union_promotes_interior, UNION_DENSE_CARD};

/// What one output key becomes: the container type the kernel will build, the
/// arena slot bytes that hold every state the fold passes through, and — for a
/// run result — the run count that bound assumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Choice {
    pub typ: u8,
    pub cap: u32,
    pub runs: u16,
}

/// Slot ceiling for a key that can only *shrink* from a seed of `card` values
/// (AND and DIFF): array form while it fits, else a bitmap.
#[inline]
pub(crate) fn shrink_cap(card: u32) -> u32 {
    if card as usize <= ARRAY_MAX_SIZE {
        card * 2
    } else {
        BITMAP_BYTES as u32
    }
}

/// AND at one key. The result is a subset of the smallest input there, so the
/// slot is sized from `min_card`; the type follows the accumulator the kernel
/// picks (`intersect_key`): an array while it fits, native runs when every
/// input is run-encoded and they stay under the run ceiling, else a bitmap.
#[inline]
pub(crate) fn intersect_key(min_card: u32, all_run: bool, total_runs: usize) -> Choice {
    let (typ, runs) = if min_card as usize <= ARRAY_MAX_SIZE {
        (CT_ARRAY, 0)
    } else if all_run && total_runs <= MAX_RUNS {
        (CT_RUN, total_runs as u16)
    } else {
        (CT_BITMAP, 0)
    };
    Choice { typ, cap: shrink_cap(min_card), runs }
}

/// Which union promotion rule applies — they differ deliberately, so the choice
/// is explicit at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fanin {
    /// A flat, one-shot union: promote on the measured break-even.
    Flat,
    /// A union inside a tree, whose result a parent fold consumes. Conservative:
    /// a bitmap intermediate poisons downstream ANDs.
    Interior,
}

/// Merged per-key statistics for a union.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnionKey {
    /// Σ cardinality over the inputs present at this key (saturating).
    pub sum_card: u32,
    /// Σ run count over those inputs.
    pub total_runs: usize,
    /// Any input is already a bitmap.
    pub any_bitmap: bool,
    /// Every input is run-encoded.
    pub all_run: bool,
    /// Largest single input's stored bytes — the slot must hold a lone copy.
    pub max_single: u32,
    /// How many inputs are present (the promotion predicate's fan-in).
    pub n: usize,
}

/// OR at one key. Mirrors `union_key`'s accumulator choice: a bitmap when any
/// input already is one, when the merged cardinality outgrows an array, when
/// the runs outgrow a bitmap, or when the fan-in makes scatter-then-count
/// cheaper than repeated merges; otherwise an array whose slot must also hold
/// the coalesced-run and lone-copy forms.
#[inline]
pub(crate) fn union_key(k: &UnionKey, fanin: Fanin) -> Choice {
    // Order matters: the three cheap tests short-circuit before the fitted
    // promotion predicate, which costs a multiply.
    let needs_bitmap = k.any_bitmap
        || k.sum_card > UNION_DENSE_CARD
        || k.total_runs > MAX_RUNS
        || match fanin {
            Fanin::Flat => union_promotes(k.n, k.sum_card),
            Fanin::Interior => union_promotes_interior(k.n, k.sum_card),
        };

    if !needs_bitmap {
        let cap = (k.sum_card as usize * 2)
            .max(run_bytes(k.total_runs))
            .max(k.max_single as usize) as u32;
        return Choice { typ: CT_ARRAY, cap, runs: 0 };
    }
    // Run ∪ Run coalesces natively and stays a run container, but it is
    // accumulated in a bitmap-sized slot so either form fits.
    if k.all_run && k.n > 0 && k.total_runs <= MAX_RUNS {
        return Choice { typ: CT_RUN, cap: BITMAP_BYTES as u32, runs: k.total_runs as u16 };
    }
    Choice { typ: CT_BITMAP, cap: BITMAP_BYTES as u32, runs: 0 }
}

/// The LHS container at one key of a difference.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DiffLhs {
    pub card: u32,
    pub typ: u8,
    pub runs: u16,
    /// Bytes the LHS occupies stored as-is — what an untouched key costs.
    pub stored: u32,
}

/// Slot ceiling for DIFF at one key. A key no subtrahend touches is copied
/// verbatim; a key some subtrahend touches can only shrink from the LHS.
///
/// Takes just the two fields it needs, so the cursor-driven planner — which
/// only wants a capacity — never pays to read a container's run count.
#[inline]
pub(crate) fn diff_cap(card: u32, stored: u32, in_rhs: bool) -> u32 {
    if in_rhs {
        shrink_cap(card)
    } else {
        stored
    }
}

/// DIFF at one key, for the tree analyzer — which only knows that the RHS
/// *shape* spans this key, not that the RHS *arena* will actually hold a
/// container there. When it doesn't, `diff_key` copies the LHS verbatim, so the
/// slot must hold whichever of the two is larger.
///
/// `splittable` is "every subtrahend here is a run or array" and `run_bound` the
/// worst-case fragment count, mirroring `diff_run_bound`.
#[inline]
pub(crate) fn diff_key_shape(
    lhs: &DiffLhs,
    in_rhs: bool,
    splittable: bool,
    run_bound: usize,
) -> Choice {
    let verbatim = Choice { typ: lhs.typ, cap: lhs.stored, runs: lhs.runs };
    if !in_rhs {
        return verbatim;
    }
    let shrunk = diff_cap(lhs.card, lhs.stored, true);
    let (typ, runs) = if lhs.typ == CT_RUN
        && lhs.card as usize > ARRAY_MAX_SIZE
        && splittable
        && run_bound <= MAX_RUNS
    {
        (CT_RUN, run_bound as u16)
    } else if lhs.card as usize <= ARRAY_MAX_SIZE {
        (CT_ARRAY, 0)
    } else {
        (CT_BITMAP, 0)
    };
    if verbatim.cap >= shrunk {
        verbatim
    } else {
        Choice { typ, cap: shrunk, runs }
    }
}
