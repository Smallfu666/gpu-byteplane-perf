#!/usr/bin/env python3
"""Recompute every number quoted in reliability/README.md from the result CSVs."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def r1() -> int:
    rows = []
    for p in sorted((ROOT / "r1_single_replica").glob("r1_*_*.csv")):
        if "aggregate" not in p.name:
            rows += read(p)
    print("== R1: single replica, CPU evaluator ==")
    print(f"cells: {len(rows)}")
    print(f"outcomes: {dict(Counter(r['outcome'] for r in rows))}")
    print(f"detected: {sum(r['detected'] == 'True' for r in rows)}/{len(rows)}")
    print(f"contains_truth: {sum(r['contains_truth'] == 'True' for r in rows)}/{len(rows)}")
    print(f"planes detected per cell: {sorted(set(r['detected_planes'] for r in rows))}")
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        ratio = {float(r["bound_width"]) / float(r["clean_answer"]) for r in sub}
        rel_err = [abs(float(r["delivered_answer"]) - float(r["clean_answer"])) / float(r["clean_answer"]) for r in sub]
        print(f"  {ds}: n_rows={sub[0]['n_rows']} radius/answer={min(ratio):.2f}..{max(ratio):.2f} "
              f"actual |error|/answer max={max(rel_err):.2e}")
    return len(rows)


def r2() -> int:
    d = ROOT / "r2_structured_faults"
    m = read(d / "r2_matrix.csv")
    o = read(d / "r2_cpu_gpu_oracle_comparison.csv")
    print("\n== R2: structured faults, H200 GPU path vs CPU oracle ==")
    print(f"configurations: {len(m)}")
    print(f"GPU outcomes: {dict(Counter(r['verdict_cell'] for r in m))}")
    print(f"CPU/GPU classification match: {sum(r['classification_match'] == 'true' for r in o)}/{len(o)}")
    print(f"silent_wrong rows: {sum(r['silent_wrong'] == 'true' for r in m)}, "
          f"cert_bound_failure: {sum(r['cert_bound_failure'] == 'true' for r in m)}, "
          f"hard_fail: {sum(r['hard_fail'] == 'true' for r in m)}")
    print("per policy (vote_recovered / bounded_degraded, out of 42):")
    pol = defaultdict(Counter)
    for r in m:
        pol[r["policy"]][r["verdict_cell"]] += 1
    for k, v in pol.items():
        print(f"  {k:<34s} {v['vote_recovered']:>2d} / {v['bounded_degraded']:>2d}")
    print("bounded cells by faulted planes and radius/answer:")
    b = Counter()
    for r in m:
        if r["verdict_cell"] == "bounded_degraded":
            norm = float(r["bound_width_norm"])
            b[(r["dataset"], r["affected_planes"], "radius < answer" if norm < 1 else "radius >= answer")] += 1
    for k, v in sorted(b.items()):
        print(f"  {k}: {v}")
    cesm = {r["bound_width"] == r["bound_width_norm"] for r in m
            if r["dataset"] == "cesm_atm_cloud" and r["verdict_cell"] == "bounded_degraded"}
    print(f"cesm_atm_cloud slice has clean answer 0 (norm == radius in every bounded row): {cesm == {True}}")
    return len(m)


def storage() -> None:
    print("\n== Storage: plane copies for an 8-plane column ==")
    for name, rv in {"naive / striping r3": [3, 1, 1, 1, 1, 1, 1, 1], "graded_nmr_B11": [3, 2, 1, 1, 1, 1, 1, 1],
                     "uniformish_B11": [2, 1, 2, 1, 2, 1, 1, 1], "fallback (r=1)": [1] * 8}.items():
        print(f"  {name:<20s} {sum(rv)} planes, +{(sum(rv) - 8) / 8:.1%}")


if __name__ == "__main__":
    a = r1()
    b = r2()
    storage()
    print(f"\nTotal injected-fault configurations: {a} + {b} = {a + b}")
