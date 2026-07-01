# frostbit

Frozen, mmap-friendly, zero-copy roaring bitmaps with fast set operations.

A *frozen* bitmap is a roaring bitmap serialized into a compact, aligned byte
buffer that can be queried **directly from raw bytes** (e.g. an `mmap`) with no
deserialization — the read-optimized counterpart to a mutable
[`roaring::RoaringBitmap`](https://docs.rs/roaring).

```rust
use frostbit::{FrozenBitmapBuilder, FrozenBitmapView, intersect_fast};

let mut b = FrozenBitmapBuilder::new();
b.extend_sorted([10, 20, 70_000]);
let bm = b.finish();              // compact, ready to persist or mmap

let v = FrozenBitmapView::from_bytes(bm.as_bytes()).unwrap();
assert!(v.contains(20));
assert_eq!(v.len(), 3);
```

## Highlights

- **Zero-copy reads.** `contains`, `rank`, `min`/`max`, and iteration run over the
  raw bytes; no allocation, no parse.
- **Two on-the-wire encodings**, chosen automatically by size: a standard
  container format (array / bitmap / run containers) and a compact inline format
  for small, scattered sets.
- **Statically-planned set operations.** `intersect_fast`, `union_fast`, and
  `difference_fast` size their working arena from an up-front analysis pass that
  proves each output container's capacity — so execution **never allocates a
  container at runtime** (no grow, no realloc). Results **serialize in place**:
  the arena reserves header room up front and payloads compact leftward, so the
  fold writes directly into what becomes the output buffer, and all working +
  result buffers cycle through per-thread pools (zero steady-state mallocs).
- **Boolean expression trees.** `BitmapExpr` combines leaves with AND / OR / DIFF
  and evaluates them from a fold plan built **once** at construction: same-op
  chains flatten to one N-way op, intermediates chain as pooled arenas
  (serialized only at the end), and a `materialize` allocates *only its result*.
  Opt-in **hole-punching** (`punch_holes`) prunes dead 64K blocks before folding,
  and an empty AND/DIFF subtree **short-circuits** the rest of the tree.
- **SIMD container kernels** (NEON / SSE2 with scalar fallbacks) plus
  autovectorized word operations.
- **First-class `roaring` interop** behind the default `roaring` feature:
  `FrozenBitmap::from_roaring` / `to_roaring` and `From` conversions.

## Status

The builder, zero-copy view, roaring conversions, flat N-way
`intersect`/`union`/`difference` ops, the `BitmapExpr` expression-tree evaluator
(plan reuse, hole-punching, subtree short-circuit), and the `_compact` op
finalizers are implemented and extensively tested — differential against
`roaring` (including 10M-element and randomized stress suites, and an
11.6k-case arena-sizing stress). API surface is still pre-1.0.

## Benchmarks

Measured with [criterion](https://docs.rs/criterion) against `roaring` 0.11.4
(with its nightly `simd` feature enabled) on Apple Silicon (aarch64 / NEON):

```
cargo +nightly bench --features roaring-simd
```

`roaring` is the mutable `RoaringBitmap`; it has no reusable plan, so it
re-evaluates on each call, whereas a `BitmapExpr` fold plan is built once. Tree
benchmarks **include plan construction** in the timed region. Numbers are
criterion medians — indicative, and vary with hardware and run.

### Expression trees

Realistic boolean filter shapes over a mixed-container leaf pool:

| shape | frostbit | roaring |
|---|---:|---:|
| `conj5` — nested ANDs, flattened to one 5-way | **41 µs** | 87 µs |
| `filter` — `base ∩ OR(domains) ∩ (¬lang)` | **28 µs** | 44 µs |
| `cnf3` — AND of OR-groups | **87 µs** | 100 µs |
| `dnf` — OR of AND-groups | **56 µs** | 78 µs |

Across randomly-generated trees (2–45 leaves) frostbit wins **every** shape,
by 1.2× up to ~3.4× — e.g. a 31-leaf tree in **141 µs** vs roaring's 476 µs,
an 18-leaf in **89 µs** vs 267 µs.

### Query-shape optimizations

| | frostbit | roaring |
|---|---:|---:|
| **Hole-punching** — narrow filter ∩ wide OR-groups (dead blocks skipped) | **8.3 µs** *(713 µs un-punched)* | 371 µs |
| **Short-circuit** — AND with an empty subtree (sibling never evaluated) | **264 ns** | 401 µs |

### N-way flat ops (8-way)

frostbit / roaring, in µs; **bold** is faster:

| | sparse arrays | dense bitmaps |
|---|---|---|
| `intersect` | **18** / 25 | **13** / 22 |
| `union` | **114** / 795 | **21** / 40 |
| `difference` | **77** / 80 | **56** / 159 |

frostbit wins all three ops on sparse **and** dense inputs, and on **run
containers** across the board (8-way union 4.0 µs vs 17 µs; difference 2.6 µs vs
4.6 µs — runs are folded natively). Balanced sorted-array intersect / union /
difference each use a SIMD shuffle-merge (CRoaring-style compare-all-pairs and
Inoue–Taura rotate networks, NEON + x86); heavily-skewed pairs dispatch to a
galloping search. Across the full 2–16-way sweep the only remaining trails are
within a few percent (16-way run-intersect, 2-way sparse-array difference).

## Features

- `roaring` *(default)* — conversions to/from `roaring::RoaringBitmap`.
- `tracing` — opt-in trace logs for parse failures.
- `internals` — exposes internal modules for white-box tests/benchmarks; not a
  stable API.

## License

MIT OR Apache-2.0.
