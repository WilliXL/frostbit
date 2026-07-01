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
use crate::ops::plan::{Op, Plan, SlotPlan, UNION_DENSE_CARD};
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
            CT_RUN => 2 + self.runs as u32 * 4,
            _ => self.card * 2,
        }
    }
}

/// A node's output shape: container metadata, ascending by key.
pub type Shape = Vec<Meta>;

/// Exact shape of a leaf, read from its container index (inline → array form).
pub fn view_shape(view: &FrozenBitmapView<'_>) -> Shape {
    let mut c = ContainerCursor::new(view);
    let mut out = Vec::new();
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

/// Slot ceiling for an AND/DIFF output seeded from `card` values (array while it
/// fits, else bitmap) — matches `plan`'s `shrink_slot_bytes`.
#[inline]
fn shrink(card: u32) -> u32 {
    if card as usize <= ARRAY_MAX_SIZE {
        card * 2
    } else {
        BITMAP_BYTES as u32
    }
}

/// Derive the arena [`Plan`] from an output shape.
pub fn to_plan(op: Op, shape: &Shape) -> Plan {
    Plan {
        op,
        slots: shape.iter().map(|m| SlotPlan { key: m.key, capacity: m.cap }).collect(),
        scratch_bytes: 2 * BITMAP_BYTES,
        double: op == Op::Diff,
    }
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
    let mut out = Vec::new();
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
            let (typ, runs) = if min_card as usize <= ARRAY_MAX_SIZE {
                (CT_ARRAY, 0)
            } else if all_run && runs <= MAX_RUNS {
                (CT_RUN, runs as u16)
            } else {
                (CT_BITMAP, 0)
            };
            out.push(Meta { key, cap: shrink(min_card), card: min_card, runs, typ });
        }
    }
    out
}

/// OR: union of keys; mirrors `union_key`'s bitmap/array/run choice.
pub fn union_shape(inputs: &[Shape]) -> Shape {
    let mut curs: Vec<Cur> = inputs.iter().map(Cur::new).collect();
    let mut out = Vec::new();
    while let Some(key) = min_key(&curs) {
        let (mut sum, mut runs, mut any_bitmap, mut all_run, mut max_single, mut n) =
            (0u32, 0usize, false, true, 0u32, 0usize);
        for c in &mut curs {
            if c.key() == Some(key) {
                let m = c.take();
                sum = sum.saturating_add(m.card);
                runs += m.runs as usize;
                any_bitmap |= m.typ == CT_BITMAP;
                all_run &= m.typ == CT_RUN;
                max_single = max_single.max(m.stored());
                n += 1;
            }
        }
        let needs_bitmap = any_bitmap || sum > UNION_DENSE_CARD || runs > MAX_RUNS;
        let (cap, typ, oruns) = if needs_bitmap && all_run && n > 0 && runs <= MAX_RUNS {
            (BITMAP_BYTES as u32, CT_RUN, runs as u16)
        } else if needs_bitmap {
            (BITMAP_BYTES as u32, CT_BITMAP, 0)
        } else {
            ((sum * 2).max(2 + runs as u32 * 4).max(max_single), CT_ARRAY, 0)
        };
        out.push(Meta { key, cap, card: sum, runs: oruns, typ });
    }
    out
}

/// DIFF: `inputs[0]` minus the rest; output keys = LHS keys (RHS only shrinks).
pub fn diff_shape(inputs: &[Shape]) -> Shape {
    let mut out = Vec::new();
    let Some((lhs, rest)) = inputs.split_first() else { return out };
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
        // The rhs *shape* over-approximates its keys, so even an `in_rhs` key may
        // be absent in the rhs *arena* at runtime — in which case `diff_key`
        // copies the lhs verbatim. The slot must hold whichever the kernel does:
        // the verbatim lhs, or the in-rhs (shrunk / split) result. Mirror
        // diff_key's type choice for the in-rhs result.
        let verbatim = l.stored();
        let (irc, irtyp, irruns) = if !in_rhs {
            (verbatim, l.typ, l.runs)
        } else if l.typ == CT_RUN && l.card as usize > ARRAY_MAX_SIZE && all_split && bound <= MAX_RUNS {
            (shrink(l.card), CT_RUN, bound as u16)
        } else if l.card as usize <= ARRAY_MAX_SIZE {
            (shrink(l.card), CT_ARRAY, 0)
        } else {
            (BITMAP_BYTES as u32, CT_BITMAP, 0)
        };
        let m = if verbatim >= irc {
            Meta { key: l.key, cap: verbatim, card: l.card, runs: l.runs, typ: l.typ }
        } else {
            Meta { key: l.key, cap: irc, card: l.card, runs: irruns, typ: irtyp }
        };
        out.push(m);
    }
    out
}
