# frostbit

Frozen, mmap-friendly, zero-copy roaring bitmaps with fast set operations.

A *frozen* bitmap is a roaring bitmap serialized into a compact, aligned byte
buffer that can be queried directly from raw bytes — an `mmap`, a network
payload — with no deserialization step. It is the read-optimized counterpart to
a mutable [`roaring::RoaringBitmap`](https://docs.rs/roaring).

```rust
use frostbit::FrozenBitmapBuilder;

let mut b = FrozenBitmapBuilder::new();
b.extend_sorted([10, 20, 70_000]);
let bm = b.finish();              // compact, ready to persist or mmap

let v = bm.view();                // zero-copy; infallible on an owned bitmap
assert!(v.contains(20));
assert_eq!(v.len(), 3);
```

## Overview

- **Zero-copy reads.** `contains`, `min`/`max`, and iteration run over the raw
  bytes without allocating or parsing.
- **Two wire encodings**, selected automatically by size: a standard container
  format (array / bitmap / run) and a compact inline format for small scattered
  sets.
- **Statically planned set operations.** `intersect_fast`, `union_fast`, and
  `difference_fast` size their working arena from an analysis pass that proves
  each output container's capacity, so execution never allocates a container at
  runtime. Results serialize in place: the arena reserves header room up front
  and payloads compact leftward, so a fold writes directly into what becomes the
  output buffer.
- **Boolean expression trees.** `BitmapExpr` combines leaves with AND / OR /
  DIFF and evaluates from a fold plan built once at construction. Same-op chains
  flatten into a single N-way operation, intermediates chain as pooled arenas,
  and `materialize` allocates only its result.
- **Query-shape optimizations.** Hole-punching prunes dead 64K blocks beneath
  any narrowing AND; an empty AND or DIFF subtree short-circuits the remainder
  of the tree. Both are derived by the analyzer and require no user action.
- **SIMD container kernels** — NEON on aarch64; SSE2 / SSSE3 / SSE4.1 / AVX2 /
  AVX-512 on x86-64 with runtime feature detection; scalar fallbacks throughout.
- **`roaring` interop** behind the default `roaring` feature:
  `FrozenBitmap::from_roaring` / `to_roaring` and `From` conversions.

## Usage

### Building

Values are pushed in strictly ascending order. The builder finalizes each 64K
block as it completes and selects the smallest payload for that block:

```rust
let mut b = FrozenBitmapBuilder::new();
b.push(3);
b.extend_sorted([70_000, 1 << 20]);
let bm = b.finish();            // smallest encoding overall (may be inline)
```

With the `roaring` feature, `FrozenBitmap::from_roaring(&rb)` and
`bm.to_roaring()` convert to and from `roaring::RoaringBitmap`.

### Reading

An owned `FrozenBitmap` is valid by construction, so `bm.view()` is infallible.
Validation exists only at the trust boundary: `FrozenBitmapView::from_bytes`
validates foreign bytes and borrows them without copying, returning `None` for
malformed input.

```rust
let Some(v) = FrozenBitmapView::from_bytes(&mapped) else {
    anyhow::bail!("corrupt bitmap segment");
};
```

`FrozenBitmap::from_bytes` is the owned, copying variant. Queries are
`contains`, `len`, `is_empty`, `min`, `max`, `num_containers`, and an ascending
`iter()`.

### Set operations

Flat N-way folds take a slice of views and produce an owned result in one pass:

```rust
let out = intersect_fast(&[a.view(), b.view(), c.view()]);
```

`_fast` results are in op-ready standard container form, suited to feeding the
next operation. The matching `_compact` variants — `intersect_compact`,
`union_compact`, `difference_compact` — return the smallest form, suited to
storage. Both fold identically and differ only in how the result serializes.

For boolean trees, build a `BitmapExpr`. Construction performs the analysis, so
build once and `materialize()` per query:

```rust
use frostbit::BitmapExpr;

let expr = BitmapExpr::and([
    BitmapExpr::leaf(base.view()),
    BitmapExpr::or([BitmapExpr::leaf(d0.view()), BitmapExpr::leaf(d1.view())]),
    BitmapExpr::difference(BitmapExpr::leaf(all.view()), BitmapExpr::leaf(lang.view())),
]);

let result = expr.materialize();  // `expr` is reusable across evaluations
```

### Working memory

By default there is nothing to configure. Op arenas, fold cursors, the operand
stack, and result buffers all live in per-thread pools. The first call on a
thread allocates; subsequent calls take, fill, and return the same buffers, so a
steady-state operation performs no allocation.

`frostbit::pool` provides an explicit bound when one is wanted:

```rust
use frostbit::pool::{self, OnOverflow, PoolConfig};

pool::configure(PoolConfig::new().buffers(16).buffer_bytes(1 << 20));
pool::prewarm();                       // allocate the budget up front

// Or give the buffers an explicit shape:
pool::configure(PoolConfig::new().buffer_sizes([4 << 20, 1 << 20, 1 << 20]));

// A fold exceeding the budget allocates a temporary and drops it on release
// (default), or fails loudly:
pool::configure(PoolConfig::new().buffers(4).on_overflow(OnOverflow::Fail));

let s = pool::stats();                 // live / retained / retained_bytes / overflows
pool::clear();                         // release memory when a worker idles
```

Budgets are per-thread, so total working memory is bounded by
`threads × budget` with no shared state or contention. Under a thread pool,
configure and pre-warm in the worker-start hook — `rayon`'s `start_handler` or
`tokio`'s `on_thread_start`.

### Wire format

Little-endian, two self-identifying encodings (v3):

```text
standard ("FROZ")                          inline ("FI")
 0  u32  MAGIC "FROZ"                       0  [u8;2] MAGIC "FI"
 4  u16  VERSION (3)                        2  u16    count
 6  u16  FLAGS (has-runs, full, has-bitmap) 4  ..     packed ascending u32s
 8  u32  NUM_CONTAINERS
12  u32  CARDINALITY (flag ⇒ 2^32)
16  ..   container index (SoA, 8 B/container: key, type, cardinality, offset)
    ..   data section
```

A container holds one 64K block — the key is the value's high 16 bits — as a
sorted-`u16` **array**, an 8 KiB **bitmap**, or a **run** payload
(`count + (start, len)` pairs), whichever is smallest. Bitmap payloads are
64-byte aligned relative to the buffer start. The inline format covers small
scattered sets where per-container overhead would dominate.

### File layout

A frozen bitmap is one contiguous, position-independent byte blob.

- **One per file:** write `bm.as_bytes()`, then `mmap` and read with
  `FrozenBitmapView::from_bytes(&map)`.
- **Many per file:** pack blobs back-to-back with a `(offset, len)` directory of
  your own, using `bm.byte_len()` at write time. Start each blob at a
  64-byte-aligned offset so interior bitmap payloads keep their alignment inside
  the mapping.

Output bytes are deterministic for a given set and views never write, so
mappings can be shared read-only across processes.

## Benchmarks

Figures are speedups relative to `roaring` 0.11 built with its nightly-only
`simd` feature — its strongest configuration. Measured with
[criterion](https://docs.rs/criterion) on Apple M4 Pro (aarch64 / NEON), medians
from a single run. `./benchmarks/run.sh` runs the suite and
`benchmarks/report.py` renders the tables.

Two notes on interpretation. Tree benchmarks include plan construction in the
timed region, so corpus figures reflect what a query engine pays per query. And
frostbit returns a serialized `FrozenBitmap` where `roaring` returns its
in-memory structure; on workloads where the fold itself is trivial that
serialization dominates the measurement, and asking `roaring` for the same
artifact reverses the result. The figures below are the unadjusted comparison.

### Expression trees

| shape | speedup |
|---|---:|
| `conj5` — nested ANDs, flattened to one 5-way | 3.8× |
| `filter` — `base ∩ OR(domains) ∩ (¬lang)` | 2.4× |
| `dnf` — OR of AND-groups | 2.1× |
| `cnf3` — AND of OR-groups | 1.2× |

Randomly generated trees of 2–38 leaves range from 1.7× to 19×.

### Tree corpus

50,000 deterministic random trees, 507,652 leaves, spanning deep AND-chains,
wide flat ORs, diff-heavy filters, and ten additional shape families:

| | speedup |
|---|---:|
| whole corpus, analysis included | 2.4× |

Analysis accounts for 11% of frostbit's corpus time; the rest is execution.

### Work-skipping

Blocks that cannot reach the result are pruned beneath any narrowing AND, and
evaluation stops as soon as an AND or DIFF operand is known empty. Both fall out
of the fold plan and need no user action.

| tree | speedup |
|---|---:|
| `narrow ∩ (w₁ ∪ w₂) ∩ (w₃ ∪ w₄)` — 4 live blocks, 256 per `w` | 209× |
| `(a \ a) ∩ (b ∪ c ∪ d)` — left operand empty, right never evaluated | >1000× |

### Flat N-way operations (8-way)

| 8-way | speedup |
|---|---:|
| intersect · sparse arrays | 1.8× |
| intersect · dense bitmaps | 1.7× |
| intersect · run containers | 2.9× |
| union · sparse arrays | 1.3× |
| union · dense bitmaps | 1.1× |
| union · run containers | 1.7× |
| difference · sparse arrays | 2.1× |
| difference · dense bitmaps | 6.5× |
| difference · run containers | 1.1× |

N-way baselines use `roaring`'s
[`MultiOps`](https://docs.rs/roaring/latest/roaring/trait.MultiOps.html) trait,
its documented fast path for merging many bitmaps.

## Repository layout

```text
src/
  format.rs        wire format constants and byte primitives
  container.rs     container payload access (array / bitmap / run / inline)
  api/
    builder.rs     ascending-order builder
    view.rs        zero-copy reader (FrozenBitmapView, Iter)
    bitmap.rs      owned FrozenBitmap
    expr.rs        BitmapExpr tree engine: plan construction and execution
    convert.rs     roaring::RoaringBitmap interop
    pool.rs        per-thread working-memory pools
  ops/
    analyze/
      plan.rs      cursor-driven planning for flat ops
      shape.rs     shape propagation for trees
      decide.rs    per-key container form and capacity decisions
    kernels/       intersect / union / difference / run / accumulators
    simd/
      intersect/   array_intersect, bitmap_intersect
      union/       array_union, bitmap_union
      difference/  array_difference, bitmap_difference
      common/      block, compact, fold, popcount, window_scan, words
    arena.rs       pooled working arena and in-place serialization
    cursor.rs      per-container cursor over leaves and arenas
    keymask.rs     hole-punch live-key mask
    source.rs      the Inputs abstraction the kernels fold over
tests/             differential and stress suites against a roaring oracle
benchmarks/        criterion benches, corpus audit, heap profiling
```

## Status

Pre-1.0; the API surface may change. The builder, zero-copy view, roaring
conversions, flat N-way operations, the `BitmapExpr` evaluator, and in-place
compact serialization are implemented and tested differentially against
`roaring`, including 10M-element round-trips, randomized stress suites, an
arena-sizing stress suite, and a 50,000-tree corpus checked for parity.

## Features

- `roaring` *(default)* — conversions to and from `roaring::RoaringBitmap`.
- `internals` — exposes internal modules for white-box tests and benchmarks;
  not a stable API.

## License

MIT OR Apache-2.0.
