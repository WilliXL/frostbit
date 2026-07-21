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
  **Hole-punching** prunes dead 64K blocks before folding — derived
  automatically whenever the tree's root provably narrows a child
  (`punch_holes()` forces it) — and an empty AND/DIFF subtree
  **short-circuits** the rest of the tree.
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
.punch_holes();                  // automatic for narrowing ANDs; this forces it

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
bench; the 25k-tree corpus at 10 × 20 s) on Apple Silicon (aarch64 / NEON).
`./benchmarks/run.sh` runs the whole suite against **both** `roaring` 0.11.4
variants — the default build (what a stable-toolchain user gets) and its
nightly-only `simd` feature — then re-measures the frostbit cells in a
dedicated third pass (same nightly build; sub-µs cells drift up to +70% inside
the 20-minute interleaved passes from ambient allocator noise that frostbit's
zero-alloc loops don't generate — roaring cells show no such drift and keep
their interleaved numbers), and renders combined tables via
`benchmarks/report.py`.

`roaring` is the mutable `RoaringBitmap`; it has no reusable plan, so it
re-evaluates on each call. All N-way baseline calls go through the library's
[`MultiOps`](https://docs.rs/roaring/latest/roaring/trait.MultiOps.html) trait —
its documented fast path for merging many bitmaps (lazy copy-on-write
promotion), which is dramatically faster than folding pairwise ops and is the
strongest way to drive the library. Tree benchmarks **include plan
construction** in the timed region (the 25k corpus builds, analyzes, and
materializes every tree per iteration — what a query engine pays per query).
Numbers are medians from one full run on a shared laptop (Apple M4 Pro) —
indicative, ±a few percent run to run.

### vs roaring (default build)

**Expression trees** — named shapes over a mixed-container leaf pool:

| shape | frostbit | roaring |
|---|---:|---:|
| `conj5` — nested ANDs, flattened to one 5-way | **18 µs** | 264 µs |
| `filter` — `base ∩ OR(domains) ∩ (¬lang)` | **16 µs** | 41 µs |
| `dnf` — OR of AND-groups | **51 µs** | 613 µs |
| `cnf3` — AND of OR-groups | **71 µs** | 106 µs |

Randomly-generated trees (2–38 leaves) win 1.5×–6.5× across the board.

**25,000-tree corpus** — deterministic random trees up to 100 leaves and 15
levels deep (per-tree shape profiles: deep AND-chains, flat 100-way ORs,
diff-heavy filters; 100 trees are guaranteed 100-leaf *and* 15-deep;
295,455 leaves total):

| | frostbit | roaring |
|---|---:|---:|
| whole corpus, per-query analysis included | **2.40 s** (10.4K trees/s) | 6.79 s (3.7K trees/s) |

**Query-shape optimizations:**

| | frostbit | roaring |
|---|---:|---:|
| **Hole-punching** — narrow filter ∩ wide OR-groups | **7.4 µs** *(241 µs un-punched)* | 1.42 ms |
| **Short-circuit** — AND with an empty subtree | **147 ns** | 705 µs |

**N-way flat ops (8-way):**

| 8-way | frostbit | roaring |
|---|---:|---:|
| intersect · sparse arrays | **16 µs** | 77 µs |
| intersect · dense bitmaps | **13 µs** | 20 µs |
| intersect · run containers | **1.2 µs** | 2.7 µs |
| union · sparse arrays | **117 µs** | 142 µs |
| union · dense bitmaps | **20 µs** | 24 µs |
| union · run containers | **3.0 µs** | 4.7 µs |
| difference · sparse arrays | **71 µs** | 916 µs |
| difference · dense bitmaps | **19 µs** | 131 µs |
| difference · run containers | **2.0 µs** | 2.1 µs |

Across the full 2–16-way sweep frostbit wins **34 of 36 cells outright and
loses none**: every sparse-array cell (1.2×–12.9×), every intersection cell
(1.06×–6.0×), every union cell, and every dense-difference cell — a word-op
roofline audit (`membw` bench group) found the andnot kernel latency-bound
and the bitmap→array extraction over-eager, and fixing both turned dense
difference from parity into **1.4×–7.0×**. The two remaining cells (run
difference at 2-/4-way) sit within ±6% and trade sides run to run. Flat
one-shot unions promote to bitmap accumulation on a crossover derived from
measured kernel constants (2-way sparse union 46 → 31 µs); tree-interior
unions keep the conservative rule — aggressive promotion inside trees
re-measured at cnf3 +62%, corpus +16%, and stays rejected.

### vs roaring's nightly `simd` build

The same comparisons against `roaring` with its (nightly-only, off-by-default)
`simd` feature — its strongest configuration:

| | frostbit | roaring `simd` |
|---|---:|---:|
| `conj5` / `filter` / `dnf` / `cnf3` | **18 / 16 / 51 / 71 µs** | 70 / 41 / 111 / 93 µs |
| 25k-tree corpus | **2.40 s** | 5.61 s |
| hole-punching (punched) | **7.3 µs** | 1.41 ms |
| short-circuit | **147 ns** | 715 µs |
| 8-way intersect (sparse / dense / runs) | **16 / 13 / 1.2 µs** | 28 / 20 / 2.7 µs |
| 8-way union (sparse / dense / runs) | **116 / 20 / 3.0 µs** | 141 / 24 / 4.7 µs |
| 8-way difference (sparse / dense / runs) | **71 / 19 / 2.0 µs** | 115 / 134 / 2.1 µs |

Every tree shape and the corpus go to frostbit (1.3×–6.5×). The flat-op sweep
mirrors the default-build picture: 34 of 36 cells outright and no losses —
every sparse, intersection, union, and dense-difference cell, with the same
two run-difference cells inside the noise band.

## Features

- `roaring` *(default)* — conversions to/from `roaring::RoaringBitmap`.
- `tracing` — opt-in trace logs for parse failures.
- `internals` — exposes internal modules for white-box tests/benchmarks; not a
  stable API.

## License

MIT OR Apache-2.0.
