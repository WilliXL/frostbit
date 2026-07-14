//! Static analysis pass for AND/OR/DIFF.
//!
//! Computes the exact set of output containers and a **proven-sufficient** byte
//! capacity for each, so the op arena is allocated once and execution never
//! grows or reallocates a slot. Every capacity is ≤ [`BITMAP_BYTES`] (a stored
//! container can't exceed a bitmap), so each slot fits any state the fold
//! passes through under the execution contract documented per op below.

use crate::format::*;
use crate::ops::cursor::{ContainerCursor, FoldScratch};
use crate::ops::source::Inputs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Intersect,
    Union,
    Diff,
}

/// One output container: its key and a proven byte ceiling for its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotPlan {
    pub key: u16,
    pub capacity: u32,
}

/// Output container layout for one op. `slots` are ascending by key.
#[derive(Debug, Clone)]
pub struct Plan {
    pub op: Op,
    pub slots: Vec<SlotPlan>,
    /// Fixed working scratch (a bitmap + a run double-buffer), allocated once.
    pub scratch_bytes: usize,
    /// Allocate a second slot region: the op folds partner-major and its array
    /// merges flip a per-slot side bit between the two (out ≠ in without a
    /// staging copy, while every pass streams its inputs sequentially).
    pub double: bool,
}

/// Scratch the kernels need: one bitmap accumulator + one run/bitmap temp.
const SCRATCH_BYTES: usize = 2 * BITMAP_BYTES;

mod slot_pool {
    use std::cell::RefCell;

    use super::SlotPlan;

    const MAX_POOLED: usize = 8;
    thread_local! {
        static POOL: RefCell<Vec<Vec<SlotPlan>>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn take() -> Vec<SlotPlan> {
        POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
    }

    pub(super) fn put(mut v: Vec<SlotPlan>) {
        v.clear();
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < MAX_POOLED {
                p.push(v);
            }
        });
    }
}

/// Return a one-shot plan's slot buffer to the per-thread pool. Tree plans
/// live inside a `FoldPlan` and simply never call this.
#[inline]
pub(crate) fn recycle(plan: Plan) {
    slot_pool::put(plan.slots);
}

impl Plan {
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Total slot-data bytes (each capacity padded to 8).
    pub fn data_bytes(&self) -> usize {
        self.slots
            .iter()
            .map(|s| align_up(s.capacity as usize, WORD_ALIGN))
            .sum()
    }
}

/// The single array↔bitmap boundary, shared by every op so containers are
/// canonical (array iff card ≤ this, bitmap iff above). Union *promotes* past
/// it; intersect/diff *demote* back below it (`finish_bitmap`). Keeping it equal
/// to [`ARRAY_MAX_SIZE`] (the wire-format array limit) means a seed with
/// `card ≤ this` is always an array — so the accumulator choice is type-aware by
/// construction and never extracts a bitmap.
pub(crate) const UNION_DENSE_CARD: u32 = ARRAY_MAX_SIZE as u32;

/// Fan-in-aware union promotion: with `n` containers at a key, the array path
/// streams ~`sum·(n+1)/2` elements (each merge re-streams the accumulator),
/// while a bitmap accumulator costs ~clear + scatter(sum) + popcount. Break-
/// even (measured: ~0.85ns/streamed element, ~450ns fixed + ~1.2ns/scatter):
/// never at n ≤ 3, sum ≳ 1000 at n = 4, falling as fan-in grows. The monorepo
/// promotes at a blunt sum > 256 regardless of fan-in, which trades away
/// low-fan-in shapes (cnf3-style AND-of-OR-groups) — fan-in awareness keeps
/// those as arrays.
#[inline]
pub(crate) fn union_promotes(n: usize, sum_card: u32) -> bool {
    n >= 4 && sum_card as usize * (n - 3) > 1000
}

/// Bytes a result container of `card` values takes in op-ready (`_fast`) form:
/// an array while it fits, otherwise a bitmap. The capacity contract sizes
/// every slot to be ≥ this for its result.
#[inline]
pub fn fast_container_bytes(card: u32) -> usize {
    if card as usize <= ARRAY_MAX_SIZE {
        card as usize * 2
    } else {
        BITMAP_BYTES
    }
}

/// Partner-major folds pay off while the whole accumulator set stays
/// cache-resident (each pass revisits every slot); past this footprint the
/// key-major order (one hot accumulator, partners interleaved) wins.
pub(crate) const PARTNER_MAJOR_MAX_ACC_BYTES: usize = 64 << 10;

/// Whether `op` should fold partner-major over these slots (and so wants the
/// mirror slot region for side-flipped array merges).
pub(crate) fn wants_partner_major(op: Op, slots: &[SlotPlan]) -> bool {
    matches!(op, Op::Diff | Op::Union)
        && slots.iter().map(|s| s.capacity as usize).sum::<usize>() <= PARTNER_MAJOR_MAX_ACC_BYTES
}

/// Slot ceiling when a key is seeded from a container of `card` values and can
/// only shrink (AND/DIFF): array form while it fits, else bitmap.
#[inline]
fn shrink_slot_bytes(card: u32) -> u32 {
    fast_container_bytes(card) as u32
}

/// AND: output keys = ∩ of all inputs' keys. Per key, the result ⊆ the
/// smallest-card input there, so execution seeds from it (expanding run/bitmap
/// → array when card ≤ 4096) and only shrinks. cap = `shrink_slot_bytes(min_card)`.
///
/// Two walks, picked by key-set shape: when some input is narrower (or n = 2),
/// drive by the fewest-container input and `advance_to` the rest — O(K_seed·n)
/// instead of O(K_union·n), the difference on selective conjuncts; at uniform
/// counts the flat min-scan is measurably cheaper per key.
pub fn plan_intersect<I: Inputs + ?Sized>(inputs: &I) -> Plan {
    let mut slots = slot_pool::take();
    if inputs.is_empty() {
        return Plan { op: Op::Intersect, slots, scratch_bytes: SCRATCH_BYTES, double: false };
    }
    let (mut seed, mut min_n, mut max_n) = (0, usize::MAX, 0);
    for i in 0..inputs.len() {
        let n = inputs.container_count(i);
        if n < min_n {
            (seed, min_n) = (i, n);
        }
        max_n = max_n.max(n);
    }
    let mut scratch = FoldScratch::take();
    let (cursors, _) = scratch.borrow();

    if inputs.len() == 2 || min_n < max_n {
        let mut driver = inputs.cursor(seed);
        cursors.extend((0..inputs.len()).filter(|&i| i != seed).map(|i| inputs.cursor(i)));
        while let Some(key) = driver.peek_key() {
            let mut min_card = driver.get().card;
            driver.advance();
            let present = cursors.iter_mut().all(|c| {
                let hit = c.advance_to(key);
                if hit {
                    min_card = min_card.min(c.get().card);
                }
                hit
            });
            if present {
                slots.push(SlotPlan { key, capacity: shrink_slot_bytes(min_card) });
            }
        }
    } else {
        cursors.extend((0..inputs.len()).map(|i| inputs.cursor(i)));
        loop {
            let Some(key) = min_key(cursors) else { break };
            let mut present = 0usize;
            let mut min_card = u32::MAX;
            for c in cursors.iter_mut() {
                if c.peek_key() == Some(key) {
                    present += 1;
                    min_card = min_card.min(c.get().card);
                    c.advance();
                }
            }
            if present == inputs.len() {
                slots.push(SlotPlan { key, capacity: shrink_slot_bytes(min_card) });
            }
        }
    }
    Plan { op: Op::Intersect, slots, scratch_bytes: SCRATCH_BYTES, double: false }
}

/// OR: output keys = ∪ of all inputs' keys. Per key the slot must hold the
/// union's op-ready form: a bitmap if any input is a bitmap, the merged
/// cardinality exceeds an array, or the runs exceed a bitmap; otherwise the max
/// of the merged-array, coalesced-run, and largest-single-input sizes. Every
/// union key gets a slot up front, so a fold never creates one.
pub fn plan_union<I: Inputs + ?Sized>(inputs: &I) -> Plan {
    let mut slots = slot_pool::take();
    if inputs.is_empty() {
        return Plan { op: Op::Union, slots, scratch_bytes: SCRATCH_BYTES, double: false };
    }
    let mut scratch = FoldScratch::take();
    let (cursors, _) = scratch.borrow();
    cursors.extend((0..inputs.len()).map(|i| inputs.cursor(i)));

    loop {
        let Some(key) = min_key(cursors) else { break };
        let mut sum_card = 0u32;
        let mut total_runs = 0usize;
        let mut any_bitmap = false;
        let mut max_single = 0usize;
        let mut n_present = 0usize;
        for c in cursors.iter_mut() {
            if c.peek_key() == Some(key) {
                let cr = c.get();
                sum_card = sum_card.saturating_add(cr.card);
                total_runs += cr.num_runs();
                any_bitmap |= cr.typ == CT_BITMAP;
                max_single = max_single.max(cr.stored_bytes());
                n_present += 1;
                c.advance();
            }
        }
        let needs_bitmap = any_bitmap
            || sum_card > UNION_DENSE_CARD
            || total_runs > MAX_RUNS
            || union_promotes(n_present, sum_card);
        let capacity = if needs_bitmap {
            BITMAP_BYTES
        } else {
            (sum_card as usize * 2)
                .max(2 + total_runs * 4)
                .max(max_single)
        } as u32;
        slots.push(SlotPlan { key, capacity });
    }
    Plan { op: Op::Union, double: wants_partner_major(Op::Union, &slots), slots, scratch_bytes: SCRATCH_BYTES }
}

/// DIFF: `inputs[0]` (A) minus the rest. Output keys = A's keys (the RHS can
/// only remove values within a key, never add a block). A key absent from every
/// RHS is copied verbatim (cap = its stored bytes); a key present in some RHS
/// can only shrink from A (cap = `shrink_slot_bytes(A_card)`).
pub fn plan_diff<I: Inputs + ?Sized>(inputs: &I) -> Plan {
    let mut slots = slot_pool::take();
    if inputs.is_empty() {
        return Plan { op: Op::Diff, slots, scratch_bytes: SCRATCH_BYTES, double: false };
    }
    let mut a = inputs.cursor(0);
    let mut scratch = FoldScratch::take();
    let (rhs, _) = scratch.borrow();
    rhs.extend((1..inputs.len()).map(|i| inputs.cursor(i)));

    while let Some(key) = a.peek_key() {
        let cr = a.get();
        let in_rhs = rhs.iter_mut().any(|c| c.advance_to(key));
        let capacity = if in_rhs {
            shrink_slot_bytes(cr.card)
        } else {
            cr.stored_bytes() as u32
        };
        slots.push(SlotPlan { key, capacity });
        a.advance();
    }
    Plan { op: Op::Diff, double: wants_partner_major(Op::Diff, &slots), slots, scratch_bytes: SCRATCH_BYTES }
}

/// Smallest current key across cursors, or `None` when all are exhausted.
fn min_key(cursors: &[ContainerCursor<'_>]) -> Option<u16> {
    cursors.iter().filter_map(|c| c.peek_key()).min()
}
