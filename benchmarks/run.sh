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

python3 benchmarks/report.py
