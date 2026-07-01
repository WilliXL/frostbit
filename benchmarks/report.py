#!/usr/bin/env python3
"""Render target/criterion results as combined comparison tables.

Reads every benchmark's criterion median and prints one aligned table per
group, with a column per engine (frostbit, roaring, roaring's nightly `simd`)
and frostbit's speedup over each roaring variant. Run `benchmarks/run.sh` to
populate both roaring variants first.
"""

import json
import os
import re

ENGINES = ["frostbit", "frostbit_punched", "roaring", "roaring-simd"]
GROUPS = ["intersect", "union", "diff", "tree", "holepunch", "shortcircuit"]
ROOT = "target/criterion"


def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fmt(ns):
    if ns is None:
        return "-"
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.1f} µs"
    return f"{ns / 1e6:.2f} ms"


def load():
    data = {}  # (group, row) -> {engine: median ns}
    for dirpath, _dirs, files in os.walk(ROOT):
        if os.path.basename(dirpath) != "new" or "benchmark.json" not in files:
            continue
        with open(os.path.join(dirpath, "benchmark.json")) as f:
            bench = json.load(f)
        group, fid = bench.get("group_id"), bench.get("function_id") or ""
        if group not in GROUPS:
            continue
        row, _, engine = fid.rpartition("/")
        if engine not in ENGINES:
            continue
        with open(os.path.join(dirpath, "estimates.json")) as f:
            ns = json.load(f)["median"]["point_estimate"]
        data.setdefault((group, row), {})[engine] = ns
    return data


def table(group, rows, data):
    engines = [e for e in ENGINES if any(e in data[(group, r)] for r in rows)]
    speedups = [e for e in ("roaring", "roaring-simd") if e in engines]
    header = [group] + engines + [f"vs {e}" for e in speedups]

    body = []
    for r in rows:
        d = data[(group, r)]
        fb = d.get("frostbit")
        cells = [r] + [fmt(d.get(e)) for e in engines]
        for e in speedups:
            cells.append(f"{d[e] / fb:.2f}x" if fb and e in d else "-")
        body.append(cells)

    widths = [max(len(row[i]) for row in [header] + body) for i in range(len(header))]
    line = lambda row: "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(row, widths)))
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for row in body:
        print(line(row))
    print()


def main():
    data = load()
    if not data:
        raise SystemExit(f"no results under {ROOT} — run benchmarks/run.sh first")
    for group in GROUPS:
        rows = sorted((r for (g, r) in data if g == group), key=natural)
        if rows:
            table(group, rows, data)


if __name__ == "__main__":
    main()
