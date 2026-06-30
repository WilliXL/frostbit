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
  container at runtime** (no grow, no realloc).
- **SIMD container kernels** (NEON / SSE2 with scalar fallbacks) plus
  autovectorized word operations.
- **First-class `roaring` interop** behind the default `roaring` feature:
  `FrozenBitmap::from_roaring` / `to_roaring` and `From` conversions.

## Status

Early. The builder, zero-copy view, roaring conversions, and flat N-way
`intersect`/`union`/`difference` ops are implemented and extensively tested
(differential against `roaring`, including 10M-element and randomized stress
suites). A lazy expression-tree evaluator, container-pruning ("hole-punching"),
and the `_compact` op finalizers are planned.

## Features

- `roaring` *(default)* — conversions to/from `roaring::RoaringBitmap`.
- `tracing` — opt-in trace logs for parse failures.
- `internals` — exposes internal modules for white-box tests/benchmarks; not a
  stable API.

## License

MIT OR Apache-2.0.
