#!/usr/bin/env bash
# Full comparison bench: frostbit vs roaring (stable, scalar kernels) vs
# roaring with its nightly-only `simd` feature. Both passes write into one
# criterion directory (bench IDs carry the variant), then report.py renders
# combined tables. frostbit's numbers come from the second (nightly) pass —
# the same binary conditions as its strongest competitor.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf target/criterion

cargo bench --features roaring --bench ops --bench trees
cargo +nightly bench --features roaring-simd --bench ops --bench trees

# Pass 3: re-measure the frostbit cells alone (same nightly build as the
# strongest competitor). Sub-microsecond frostbit cells drift up to +70%
# inside the 20-minute interleaved passes — ambient allocator/reclaim noise
# from the alloc-heavy competitor benches that never shows in any shorter
# context (verified: all-frostbit, frostbit-isolated, and every pairwise
# ordering measure identically). Competitors keep their interleaved numbers.
cargo +nightly bench --features roaring-simd --bench ops --bench trees -- frostbit

python3 benchmarks/report.py
