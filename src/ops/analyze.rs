//! Static analysis: what each op will produce, decided before it runs.
//!
//! Nothing here touches container payloads — it reads keys, cardinalities and
//! types, and from those proves a byte ceiling for every output slot, so the
//! arena is allocated once and execution never grows or reallocates.
//!
//! - `decide` owns the per-key rules (container type, slot capacity, run
//!   count). Both passes below route through it so they cannot drift.
//! - [`plan`] is the cursor-driven pass for a flat N-way op: it walks live
//!   inputs and emits the [`Plan`](plan::Plan).
//! - [`shape`] is the bottom-up pass for an expression tree: each node's output
//!   [`Shape`](shape::Shape) is derived from its children's, so a whole tree is
//!   analyzed once, at construction.

pub(crate) mod decide;
pub mod plan;
pub mod shape;
