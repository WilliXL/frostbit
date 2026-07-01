# frostbit

Frozen, mmap-friendly, zero-copy roaring bitmaps with fast set operations.

A *frozen* bitmap is a roaring bitmap serialized into a compact, aligned byte
buffer that can be queried **directly from raw bytes** (e.g. an `mmap`) with no
deserialization — the read-optimized counterpart to a mutable
[`roaring::RoaringBitmap`](https://docs.rs/roaring).

```rust
use frostbit::FrozenBitmapBuilder;

let mut b = FrozenBitmapBuilder::new();
b.extend_sorted([10, 20, 70_000]);
let bm = b.finish();              // compact, ready to persist or mmap

let v = bm.view();                // zero-copy view; infallible on an owned bitmap
assert!(v.contains(20));
assert_eq!(v.len(), 3);
```

## Highlights

- **Zero-copy reads.** `contains`, `min`/`max`, and iteration run over the raw
  bytes; no allocation, no parse.
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
- **SIMD container kernels** (NEON / SSE with scalar fallbacks) plus
  autovectorized word operations.
- **First-class `roaring` interop** behind the default `roaring` feature:
  `FrozenBitmap::from_roaring` / `to_roaring` and `From` conversions.

## Usage

### Building

Values go in strictly ascending order; the builder finalizes each 64K block's
container as it completes and picks the smallest payload per block:

```rust
let mut b = FrozenBitmapBuilder::new();
b.push(3);
b.extend_sorted([70_000, 1 << 20]);
let bm = b.finish();            // picks the smallest encoding (may be inline)
// or: b.finish_standard()      // always the standard container format
```

`finish()` produces the compact form for persistence. With the `roaring`
feature, `FrozenBitmap::from_roaring(&rb)` / `bm.to_roaring()` (and `From`
impls) convert to/from `roaring::RoaringBitmap` by container transcoding.

### Wire format

Little-endian, two self-identifying encodings (v3):

```text
standard ("FROZ")                          inline ("FI")
 0  u32  MAGIC "FROZ"                       0  [u8;2] MAGIC "FI"
 4  u16  VERSION (3)                        2  u16    count
 6  u16  FLAGS (has-runs, full)             4  ..     packed ascending u32s
 8  u32  NUM_CONTAINERS
12  u32  CARDINALITY (flag ⇒ 2^32)
16  ..   container index (SoA, 8 B/container: key, type, cardinality, offset)
    ..   data section
```

A container holds one 64K block (key = value's high 16 bits) as a sorted-`u16`
**array**, an 8 KiB **bitmap**, or **run** (`count + (start, len) pairs`)
payload — whichever is smallest. Bitmap payloads are 64-byte aligned relative
to the buffer start so they sit on cache lines for SIMD. The inline format
covers tiny scattered sets where per-container overhead would dominate.

### Reading

An owned `FrozenBitmap` is valid by construction, so `bm.view()` is
**infallible** — reach for it whenever you built or converted the bitmap
yourself. Validation only exists at the trust boundary: for foreign bytes
(an `mmap`, a network payload), `FrozenBitmapView::from_bytes(&[u8])`
validates and borrows without copying, returning `None` for malformed input —
handle it there rather than unwrapping:

```rust
let Some(v) = FrozenBitmapView::from_bytes(&mapped) else {
    anyhow::bail!("corrupt bitmap segment");
};
```

`FrozenBitmap::from_bytes` is the owned (copying) variant. Queries: `contains`,
`len`, `is_empty`, `min`, `max`, `num_containers`, and ascending `iter()`.

### Ops

Flat N-way folds take a slice of views and produce an owned result in one pass:

```rust
let out = intersect_fast(&[a.view(), b.view(), c.view()]);
```

`_fast` results are in op-ready standard form, ideal for feeding the next op.
For boolean *trees*, build a `BitmapExpr` — construction **is** the analysis,
so build once and `materialize()` per query:

```rust
use frostbit::BitmapExpr;

let expr = BitmapExpr::and([
    BitmapExpr::leaf(base.view()),
    BitmapExpr::or([BitmapExpr::leaf(d0.view()), BitmapExpr::leaf(d1.view())]),
    BitmapExpr::difference(BitmapExpr::leaf(all.view()), BitmapExpr::leaf(lang.view())),
])
.punch_holes();                  // opt-in: prune blocks that can't survive the AND

let result = expr.materialize(); // reuse `expr` for repeated evaluation
```

### Pre-allocation

Nothing to configure: every working buffer — op arenas, fold cursors, the
operand stack, and result buffers — lives in **per-thread pools**. The first
call on a thread allocates (sized by the op's fold plan); after that, ops take,
fill, and return the same buffers, and results serialize **in place** inside
the arena, so a steady-state op or `materialize()` performs **zero mallocs**.
Cold paths degrade gracefully: an empty pool just allocates once and the
buffer joins the cycle.

### File layout

A frozen bitmap is one contiguous, position-independent byte blob:

- **One per file:** write `bm.as_bytes()`, later `mmap` and
  `FrozenBitmapView::from_bytes(&map)`.
- **Many per file (segment):** pack blobs back-to-back with your own
  `(offset, len)` directory (`bm.byte_len()` at write time). Start each blob at
  a 64-byte-aligned offset so interior bitmap payloads keep their cache-line
  alignment inside the mapping.

The bytes are deterministic for a given set, and views never write, so
mappings can be shared read-only across processes.

## Repository layout

```text
src/
  format.rs      wire format: constants + byte primitives
  builder.rs     ascending-order builder
  view.rs        zero-copy reader (FrozenBitmapView, Iter)
  bitmap.rs      owned FrozenBitmap + per-thread result-buffer pool
  container.rs   container payload access (array/bitmap/run/inline)
  expr.rs        BitmapExpr / FoldPlan tree engine
  convert.rs     roaring::RoaringBitmap interop
  ops/
    plan.rs      static analysis: proven slot capacities per op
    shape.rs     bottom-up output-shape analysis for trees
    kernels.rs   AND/OR/DIFF fold kernels + array/run accumulators
    arena.rs     pooled working arena + in-place serialization
    cursor.rs    per-container cursor over leaves and arenas
    source.rs    the Inputs abstraction (views, arenas) the kernels fold over
    keymask.rs   hole-punching live-key mask
    run.rs       native run-container ops
    simd/        NEON / SSE kernels (merge, scan, bitmap words, popcount)
tests/           differential + stress suites (vs roaring oracle)
benchmarks/      criterion benches vs roaring
```

## Status

The builder, zero-copy view, roaring conversions, flat N-way
`intersect`/`union`/`difference` ops, the `BitmapExpr` expression-tree evaluator
(plan reuse, hole-punching, subtree short-circuit), and in-place compact
serialization are implemented and extensively tested — differential against
`roaring` (including 10M-element and randomized stress suites, and an
11.6k-case arena-sizing stress). API surface is still pre-1.0.

## Benchmarks

Measured with [criterion](https://docs.rs/criterion) (30 samples × 4 s per
bench) against `roaring` 0.11.4 with its nightly `simd` feature enabled, on
Apple Silicon (aarch64 / NEON):

```
cargo +nightly bench --features roaring-simd
```

`roaring` is the mutable `RoaringBitmap`; it has no reusable plan, so it
re-evaluates on each call, whereas a `BitmapExpr` fold plan is built once. Tree
benchmarks **include plan construction** in the timed region. Numbers are
criterion medians from one full run on a shared laptop — indicative, and they
vary a few percent run to run.

### Expression trees

Realistic boolean filter shapes over a mixed-container leaf pool:

| shape | frostbit | roaring |
|---|---:|---:|
| `conj5` — nested ANDs, flattened to one 5-way | **49 µs** | 82 µs |
| `filter` — `base ∩ OR(domains) ∩ (¬lang)` | **26 µs** | 45 µs |
| `dnf` — OR of AND-groups | **52 µs** | 79 µs |
| `cnf3` — AND of OR-groups | 105 µs | 103 µs |

Randomly-generated trees (2–38 leaves) win across the board, 1.3×–5.8×: a
4-leaf tree in **3.9 µs** vs 23 µs, an 18-leaf in **110 µs** vs 269 µs, a
31-leaf in **133 µs** vs 469 µs.

### Query-shape optimizations

| | frostbit | roaring |
|---|---:|---:|
| **Hole-punching** — narrow filter ∩ wide OR-groups (dead blocks skipped) | **7.5 µs** *(241 µs un-punched)* | 382 µs |
| **Short-circuit** — AND with an empty subtree (sibling never evaluated) | **137 ns** | 402 µs |

### N-way flat ops (8-way)

frostbit / roaring, in µs; **bold** is faster:

| | sparse arrays | dense bitmaps | run containers |
|---|---|---|---|
| `intersect` | **17** / 25 | **13** / 23 | **2.7** / 4.5 |
| `union` | **117** / 784 | **21** / 39 | **3.2** / 7.4 |
| `difference` | 85 / **81** | **56** / 153 | **2.1** / 5.1 |

frostbit wins the sweep at every arity for intersect (sparse, dense, and runs),
union (up to 6.7× at 8-way sparse), dense difference (~3–4×), and run
difference (~2.5×). Balanced sorted-array intersect / union / difference each
use a SIMD shuffle-merge (CRoaring-style compare-all-pairs and Inoue–Taura
rotate networks, NEON + x86); heavily-skewed pairs dispatch to a galloping
search. The one trail is **high-arity sparse-array difference** (8-way ~5%,
16-way varies 175–220 µs vs roaring's ~185 run-to-run): its isolated fold
profile matches roaring — the residual is machine-load variance, not kernel
work. 2-way sparse difference and `cnf3` sit within a few percent either way.

## Features

- `roaring` *(default)* — conversions to/from `roaring::RoaringBitmap`.
- `tracing` — opt-in trace logs for parse failures.
- `internals` — exposes internal modules for white-box tests/benchmarks; not a
  stable API.

## License

MIT OR Apache-2.0.
