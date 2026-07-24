//! Static analysis pass for AND/OR/DIFF.
//!
//! Computes the exact set of output containers and a **proven-sufficient** byte
//! capacity for each, so the op arena is allocated once and execution never
//! grows or reallocates a slot. Every capacity is ≤ [`BITMAP_BYTES`] (a stored
//! container can't exceed a bitmap), so each slot fits any state the fold
//! passes through under the execution contract documented per op below.

use crate::format::*;
use crate::ops::cursor::{ContainerCursor, FoldScratch};
use crate::ops::decide;
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
pub(crate) const SCRATCH_BYTES: usize = 2 * BITMAP_BYTES;

mod slot_pool {
    use super::SlotPlan;
    use crate::pool::Pool;

    thread_local! {
        static POOL: Pool<Vec<SlotPlan>> = const { Pool::new("slot-plan") };
    }

    pub(super) fn take() -> Vec<SlotPlan> {
        POOL.with(|p| p.take(Vec::new))
    }

    pub(super) fn put(mut v: Vec<SlotPlan>) {
        v.clear();
        POOL.with(|p| p.put(v));
    }

    pub(crate) fn clear() {
        POOL.with(Pool::clear);
    }
}

pub(crate) use slot_pool::clear as clear_slot_pool;

/// Return a one-shot plan's slot buffer to the per-thread pool. Tree plans
/// live inside a `FoldPlan` and simply never call this.
#[inline]
pub(crate) fn recycle(plan: Plan) {
    slot_pool::put(plan.slots);
}

/// Capacity-analysis-free plan for tiny inputs: slots = input `seed`'s keys,
/// all `B`-clamped. Sound because every ∧/∖ fold state fits a bitmap-sized
/// slot (the shrink caps only ever tighten this), and the key set is a
/// superset of the output's (unclaimed slots serialize to nothing). Not for
/// ∨, whose per-key clamp doubles as the promotion decision.
pub(crate) fn plan_trivial<I: Inputs + ?Sized>(op: Op, inputs: &I, seed: usize) -> Plan {
    debug_assert!(!matches!(op, Op::Union));
    let mut slots = slot_pool::take();
    let mut c = inputs.cursor(seed);
    while let Some(key) = c.peek_key() {
        slots.push(SlotPlan { key, capacity: BITMAP_BYTES as u32 });
        c.advance();
    }
    Plan { op, slots, scratch_bytes: SCRATCH_BYTES, double: false }
}

impl Plan {
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
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
/// streams ~`sum·f(n)` elements, `f(n) = (n+1)/2 − 1/n` (each merge re-streams
/// the accumulator), while a bitmap accumulator costs clear + scatter(sum) +
/// popcount. Break-even from measured kernel constants (merge ~0.85 ns/el,
/// clear+popcount ~149 ns, scatter ~0.43 ns/el; `benchmarks/ops.rs::membw`)
/// with a 1.35x margin on the fixed cost:
///   s · (0.85·f(n) − 0.43) > 200  ⟺  s · (85n(n+1) − 86n − 170) > 40000·n
/// i.e. s ≳ 476 at n = 2, 203 at n = 3, 135 at n = 4, falling with fan-in.
/// (An earlier rule never promoted at n ≤ 3; it was fitted to pre-audit
/// kernels whose fixed cost measured 3x higher — see the ablation ledger.)
#[inline]
pub(crate) fn union_promotes(n: usize, sum_card: u32) -> bool {
    // A lone container at a key is copied, never merged, so there is no
    // array-merge cost to trade against a bitmap — it must never promote. This
    // also guards the `coef` arithmetic, which underflows below n = 2
    // (`85·1·2 − 86 − 170`) and would overflow u64 for absurdly large n.
    if !(2..=MAX_UNION_FANIN).contains(&n) {
        return n > MAX_UNION_FANIN; // beyond the fitted range, always promote
    }
    let n = n as u64;
    let coef = 85 * n * (n + 1) - 86 * n - 170;
    sum_card as u64 * coef > 40_000 * n
}

/// Fan-in past which the promotion formula is out of its fitted range (and its
/// `u64` `coef` would eventually overflow); above it a key's union is dense
/// enough that a bitmap accumulator always wins, so promote unconditionally.
const MAX_UNION_FANIN: usize = 1 << 20;

/// Tree-interior promotion: conservative, because a bitmap intermediate is
/// consumed by a parent fold — measured on the corpus, aggressive promotion
/// poisons downstream ANDs (cnf3 +62%, corpus +16%) even though it wins the
/// flat op. Never at n ≤ 3; s(n−3) > 1000 above.
#[inline]
pub(crate) fn union_promotes_interior(n: usize, sum_card: u32) -> bool {
    n >= 4 && sum_card as usize * (n - 3) > 1000
}

/// Bytes a result container of `card` values takes in op-ready (`_fast`) form.
/// The capacity contract sizes every slot to be ≥ this for its result, which is
/// the invariant the arena-sizing stress tests assert — hence `internals`-only.
#[cfg(feature = "internals")]
#[inline]
pub fn fast_container_bytes(card: u32) -> usize {
    decide::shrink_cap(card) as usize
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


/// AND: output keys = ∩ of all inputs' keys. Per key, the result ⊆ the
/// smallest-card input there, so execution seeds from it (expanding run/bitmap
/// → array when card ≤ 4096) and only shrinks. cap = `decide::shrink_cap(min_card)`.
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
                slots.push(SlotPlan { key, capacity: decide::shrink_cap(min_card) });
            }
        }
    } else {
        cursors.extend((0..inputs.len()).map(|i| inputs.cursor(i)));
        while let Some(key) = min_key(cursors) {
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
                slots.push(SlotPlan { key, capacity: decide::shrink_cap(min_card) });
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

    while let Some(key) = min_key(cursors) {
        let mut sum_card = 0u32;
        let mut total_runs = 0usize;
        let mut any_bitmap = false;
        let mut all_run = true;
        let mut max_single = 0usize;
        let mut n_present = 0usize;
        for c in cursors.iter_mut() {
            if c.peek_key() == Some(key) {
                let cr = c.get();
                sum_card = sum_card.saturating_add(cr.card);
                total_runs += cr.num_runs();
                any_bitmap |= cr.typ == CT_BITMAP;
                all_run &= cr.typ == CT_RUN;
                max_single = max_single.max(cr.stored_bytes());
                n_present += 1;
                c.advance();
            }
        }
        let capacity = decide::union_key(
            &decide::UnionKey {
                sum_card,
                total_runs,
                any_bitmap,
                all_run,
                max_single: max_single as u32,
                n: n_present,
            },
            decide::Fanin::Flat,
        )
        .cap;
        slots.push(SlotPlan { key, capacity });
    }
    Plan { op: Op::Union, double: wants_partner_major(Op::Union, &slots), slots, scratch_bytes: SCRATCH_BYTES }
}

/// DIFF: `inputs[0]` (A) minus the rest. Output keys = A's keys (the RHS can
/// only remove values within a key, never add a block). A key absent from every
/// RHS is copied verbatim (cap = its stored bytes); a key present in some RHS
/// can only shrink from A (cap = `decide::shrink_cap(A_card)`).
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
        let capacity = decide::diff_cap(cr.card, cr.stored_bytes() as u32, in_rhs);
        slots.push(SlotPlan { key, capacity });
        a.advance();
    }
    Plan { op: Op::Diff, double: wants_partner_major(Op::Diff, &slots), slots, scratch_bytes: SCRATCH_BYTES }
}

/// Smallest current key across cursors, or `None` when all are exhausted.
fn min_key(cursors: &[ContainerCursor<'_>]) -> Option<u16> {
    cursors.iter().filter_map(|c| c.peek_key()).min()
}
