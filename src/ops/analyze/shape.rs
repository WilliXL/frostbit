//! Bottom-up output-shape analysis for tree evaluation.
//!
//! Each node's *shape* is the metadata of the containers it would produce —
//! per key: a proven slot byte-ceiling plus a bound on the output container's
//! `(card, runs, type)`. Leaf shapes are exact (read from the index); combine
//! shapes are derived from the children's shapes using the same type decisions
//! the kernels make, so the slot ceiling is always sufficient. This lets the
//! analyzer size every op's arena up front — the executor does no sizing.

use crate::format::*;
use crate::ops::cursor::ContainerCursor;
use crate::ops::analyze::decide;
use crate::ops::analyze::plan::{wants_partner_major, Op, Plan, SlotPlan, SCRATCH_BYTES};
use crate::FrozenBitmapView;

/// One output container's analysis: arena slot ceiling + parent-facing bound.
#[derive(Clone, Copy)]
pub struct Meta {
    pub key: u16,
    pub cap: u32, // proven arena slot bytes
    pub card: u32,
    pub runs: u16,
    pub typ: u8, // CT_ARRAY / CT_BITMAP / CT_RUN
}

impl Meta {
    /// Bytes this container occupies as a stored payload (parent sizing).
    #[inline]
    fn stored(&self) -> u32 {
        match self.typ {
            CT_BITMAP => BITMAP_BYTES as u32,
            CT_RUN => run_bytes(self.runs as usize) as u32,
            _ => self.card * 2,
        }
    }
}

/// A node's output shape: container metadata, ascending by key.
pub type Shape = Vec<Meta>;

/// Exact shape of a leaf, read from its container index (inline → array form).
pub fn view_shape(view: &FrozenBitmapView<'_>) -> Shape {
    let mut c = ContainerCursor::new(view);
    // Exactly one entry per container — size it once instead of regrowing.
    let mut out = Vec::with_capacity(c.container_count());
    while c.peek_key().is_some() {
        let r = c.get();
        let typ = if r.typ == CT_INLINE { CT_ARRAY } else { r.typ };
        out.push(Meta {
            key: r.key,
            cap: r.stored_bytes() as u32,
            card: r.card,
            runs: r.num_runs() as u16,
            typ,
        });
        c.advance();
    }
    out
}

/// Derive the arena [`Plan`] from an output shape.
pub fn to_plan(op: Op, shape: &Shape) -> Plan {
    let slots: Vec<SlotPlan> =
        shape.iter().map(|m| SlotPlan { key: m.key, capacity: m.cap }).collect();
    Plan { op, double: wants_partner_major(op, &slots), slots, scratch_bytes: SCRATCH_BYTES }
}

/// A merge cursor over a shape.
struct Cur<'a> {
    s: &'a Shape,
    i: usize,
}
impl<'a> Cur<'a> {
    fn new(s: &'a Shape) -> Self {
        Cur { s, i: 0 }
    }
    fn key(&self) -> Option<u16> {
        self.s.get(self.i).map(|m| m.key)
    }
    fn take(&mut self) -> Meta {
        let m = self.s[self.i];
        self.i += 1;
        m
    }
}

fn min_key(curs: &[Cur<'_>]) -> Option<u16> {
    curs.iter().filter_map(Cur::key).min()
}

/// AND: keys present in every input; result ⊆ the smallest, so cap = shrink(min).
pub fn intersect_shape(inputs: &[Shape]) -> Shape {
    let mut curs: Vec<Cur> = inputs.iter().map(Cur::new).collect();
    // At most the smallest input's key count survives an intersection.
    let mut out = Vec::with_capacity(inputs.iter().map(Vec::len).min().unwrap_or(0));
    while let Some(key) = min_key(&curs) {
        let (mut present, mut min_card, mut all_run, mut runs) = (0usize, u32::MAX, true, 0usize);
        for c in &mut curs {
            if c.key() == Some(key) {
                let m = c.take();
                present += 1;
                min_card = min_card.min(m.card);
                all_run &= m.typ == CT_RUN;
                runs += m.runs as usize;
            }
        }
        if present == inputs.len() {
            let c = decide::intersect_key(min_card, all_run, runs);
            out.push(Meta { key, cap: c.cap, card: min_card, runs: c.runs, typ: c.typ });
        }
    }
    out
}

/// OR: union of keys; mirrors `union_key`'s bitmap/array/run choice.
/// `weights[i]` is input `i`'s flattened operand count: a summarized sub-union
/// stands for that many executor operands, so the fan-in promotion predicate
/// must count it as such (an upper bound on the runtime fan-in per key).
pub fn union_shape(inputs: &[Shape], weights: &[usize]) -> Shape {
    debug_assert_eq!(inputs.len(), weights.len());
    let mut curs: Vec<Cur> = inputs.iter().map(Cur::new).collect();
    // A union spans at least the widest input's keys.
    let mut out = Vec::with_capacity(inputs.iter().map(Vec::len).max().unwrap_or(0));
    while let Some(key) = min_key(&curs) {
        let (mut sum, mut runs, mut any_bitmap, mut all_run, mut max_single, mut n) =
            (0u32, 0usize, false, true, 0u32, 0usize);
        for (c, w) in curs.iter_mut().zip(weights) {
            if c.key() == Some(key) {
                let m = c.take();
                sum = sum.saturating_add(m.card);
                runs += m.runs as usize;
                any_bitmap |= m.typ == CT_BITMAP;
                all_run &= m.typ == CT_RUN;
                max_single = max_single.max(m.stored());
                n += w;
            }
        }
        let c = decide::union_key(
            &decide::UnionKey {
                sum_card: sum,
                total_runs: runs,
                any_bitmap,
                all_run,
                max_single,
                n,
            },
            decide::Fanin::Interior,
        );
        out.push(Meta { key, cap: c.cap, card: sum, runs: c.runs, typ: c.typ });
    }
    out
}

/// DIFF: `inputs[0]` minus the rest; output keys = LHS keys (RHS only shrinks).
pub fn diff_shape(inputs: &[Shape]) -> Shape {
    let Some((lhs, rest)) = inputs.split_first() else { return Vec::new() };
    // Output keys are exactly the LHS keys.
    let mut out = Vec::with_capacity(lhs.len());
    let mut rhs: Vec<Cur> = rest.iter().map(Cur::new).collect();
    for l in lhs {
        // Gather RHS containers at this key.
        let (mut in_rhs, mut all_split, mut bound) = (false, true, l.runs as usize);
        for c in &mut rhs {
            while c.key().is_some_and(|k| k < l.key) {
                c.take();
            }
            if c.key() == Some(l.key) {
                let m = c.take();
                in_rhs = true;
                match m.typ {
                    CT_RUN => bound += m.runs as usize,
                    CT_ARRAY => bound += m.card as usize,
                    _ => all_split = false,
                }
            }
        }
        let lhs = decide::DiffLhs { card: l.card, typ: l.typ, runs: l.runs, stored: l.stored() };
        let c = decide::diff_key_shape(&lhs, in_rhs, all_split, bound);
        out.push(Meta { key: l.key, cap: c.cap, card: l.card, runs: c.runs, typ: c.typ });
    }
    out
}
