#!/usr/bin/env python3
"""Structured fault matrix (experiment R2): GPU pipeline vs CPU oracle.

For every (dataset, policy, fault family, seed) the script
  1. builds a deterministic fault plan (plane, replica, offset, XOR mask),
  2. evaluates it with a CPU oracle (vote, digest, interval, exact answer check),
  3. runs src/bench_failsoft_pipeline on the same planes and plan,
  4. records both outcomes and whether they match.

2 datasets x 5 policies x 7 fault families x 3 seeds = 210 configurations.

Example:
  python3 scripts/run_structured_faults.py \
      --gpu-bench build/reliability/bench_failsoft_pipeline \
      --planes hurricane_u=<PLANES_DIR_1> --planes cesm_atm_cloud=<PLANES_DIR_2> \
      --n-rows 500000 --out-dir out/r2

Use --cpu-only to evaluate the oracle without a GPU.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PLANE_COUNT = 8

# Replica count per plane (plane 0 = most significant byte).
POLICIES: dict[str, list[int]] = {
    # Same replica vector; they differ in how a "domain" fault lands on the
    # replicas: naive = all copies share one fault domain, striping = copies
    # are in different domains (a logical label, not a physical HBM placement).
    "naive_same_region_replication_r3": [3, 1, 1, 1, 1, 1, 1, 1],
    "logical_striping_diversity_r3": [3, 1, 1, 1, 1, 1, 1, 1],
    "graded_nmr_B11": [3, 2, 1, 1, 1, 1, 1, 1],
    "uniformish_B11": [2, 1, 2, 1, 2, 1, 1, 1],
    "certified_degradation_fallback": [1, 1, 1, 1, 1, 1, 1, 1],
}

SEEDS = [0, 1, 2]

FAULT_SCENARIOS: list[dict[str, Any]] = [
    {"scenario_id": 0, "fault_family": "single_domain_burst", "plane_label": "msb", "plane": 0,
     "burst_length": 4, "replica_mode": "policy_domain_aware"},
    {"scenario_id": 1, "fault_family": "regional_multi_byte", "plane_label": "msb", "plane": 0,
     "regional_cluster_size": 64, "replica_mode": "policy_domain_aware"},
    {"scenario_id": 2, "fault_family": "column_like_repeated_offset", "plane_label": "msb", "plane": 0,
     "column_like_stride": 4096, "repeat_count": 8, "replica_mode": "policy_domain_aware"},
    {"scenario_id": 3, "fault_family": "plane_localized_msb", "plane_label": "msb", "plane": 0,
     "burst_length": 8, "replica_mode": "single_replica"},
    {"scenario_id": 4, "fault_family": "plane_localized_lsb", "plane_label": "lsb", "plane": 7,
     "burst_length": 8, "replica_mode": "single_replica"},
    {"scenario_id": 5, "fault_family": "same_fault_all_correlated_control", "plane_label": "msb", "plane": 0,
     "burst_length": 4, "replica_mode": "same_fault_all"},
    {"scenario_id": 6, "fault_family": "mixed_independent_plus_correlated", "plane_label": "mixed", "plane": 0,
     "burst_length": 4, "replica_mode": "mixed"},
]


# ---------------------------------------------------------------- data

def available_planes(d: Path) -> list[int]:
    return [p for p in range(PLANE_COUNT) if (d / f"plane_{p:03d}.bin").is_file()]


def load_planes(d: Path, n_rows: int) -> list[np.ndarray]:
    planes = []
    for p in available_planes(d):
        a = np.frombuffer((d / f"plane_{p:03d}.bin").read_bytes()[:n_rows], dtype=np.uint8)
        if a.size < n_rows:
            raise SystemExit(f"{d}: plane {p} has {a.size} rows < --n-rows {n_rows}")
        planes.append(a)
    return planes


# ---------------------------------------------------------------- fault plans

def replicas_of(r_vector: list[int], plane: int) -> int:
    return max(r_vector[plane], 1) if plane < len(r_vector) else 1


def contiguous(rng: random.Random, n: int, length: int) -> list[int]:
    base = rng.randint(0, max(n - length, 0))
    return [base + i for i in range(length)]


def column_like(rng: random.Random, n: int, stride: int, repeat: int) -> list[int]:
    base = rng.randint(0, min(stride - 1, n - 1))
    return [base + i * stride for i in range(repeat) if base + i * stride < n]


def domain_targets(policy: str, r_vector: list[int], plane: int, rng: random.Random) -> list[int]:
    r = replicas_of(r_vector, plane)
    if r <= 1:
        return [0]
    if policy == "naive_same_region_replication_r3":
        return list(range(r))
    return [rng.randrange(r)]


def one_replica(r_vector: list[int], plane: int, rng: random.Random) -> list[int]:
    r = replicas_of(r_vector, plane)
    return [0] if r <= 1 else [rng.randrange(r)]


def fault_plan(policy: str, sc: dict[str, Any], seed: int, r_vector: list[int], n: int,
               lsb_plane: int) -> list[tuple[int, int, int, int]]:
    rng = random.Random(seed * 1000 + sc["scenario_id"])
    plane = lsb_plane if sc["plane_label"] == "lsb" else int(sc["plane"])
    fam = sc["fault_family"]
    plan: list[tuple[int, int, int, int]] = []

    def add(pl: int, reps: list[int], offs: list[int], mask: int) -> None:
        for o in offs:
            for r in reps:
                plan.append((pl, r, o, mask))

    if fam == "single_domain_burst":
        offs = contiguous(rng, n, sc["burst_length"])
        add(plane, domain_targets(policy, r_vector, plane, rng), offs, 0xFF)
    elif fam == "regional_multi_byte":
        offs = contiguous(rng, n, sc["regional_cluster_size"])
        add(plane, domain_targets(policy, r_vector, plane, rng), offs, 0xFF)
    elif fam == "column_like_repeated_offset":
        offs = column_like(rng, n, sc["column_like_stride"], sc["repeat_count"])
        add(plane, domain_targets(policy, r_vector, plane, rng), offs, 0xFF)
    elif fam in ("plane_localized_msb", "plane_localized_lsb"):
        offs = contiguous(rng, n, sc["burst_length"])
        add(plane, one_replica(r_vector, plane, rng), offs, 0xFF)
    elif fam == "same_fault_all_correlated_control":
        offs = contiguous(rng, n, sc["burst_length"])
        add(plane, list(range(replicas_of(r_vector, plane))), offs, 0xFF)
    elif fam == "mixed_independent_plus_correlated":
        offs = contiguous(rng, n, sc["burst_length"])
        add(0, list(range(replicas_of(r_vector, 0))), offs, 0xFF)
        lsb_rng = random.Random(seed * 1000 + sc["scenario_id"] + 77)
        offs2 = contiguous(lsb_rng, n, 4)
        add(lsb_plane, one_replica(r_vector, lsb_plane, lsb_rng), offs2, 0x55)
    else:
        raise ValueError(fam)
    return plan


# ---------------------------------------------------------------- CPU oracle

def vote(reps: list[np.ndarray]) -> np.ndarray:
    if len(reps) == 1:
        return reps[0]
    if len(reps) == 2:
        return np.maximum(reps[0], reps[1])
    a, b, c = reps
    return np.where((a == b) | (a == c), a, np.where(b == c, b, np.maximum(a, np.maximum(b, c))))


def cpu_oracle(clean: list[np.ndarray], r_vector: list[int], plan: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    n = clean[0].size
    reps = [[c.copy() for _ in range(replicas_of(r_vector, p))] for p, c in enumerate(clean)]
    for p, r, o, m in plan:
        reps[p][r][o] ^= m
    detected, err, radius, silent, undetected_bytes = [], 0, 0, False, 0
    for p, c in enumerate(clean):
        v = vote(reps[p])
        cs, vs = int(c.sum(dtype=np.uint64)), int(v.sum(dtype=np.uint64))
        weight = 1 << (8 * (PLANE_COUNT - 1 - p))
        err += (vs - cs) * weight
        if (vs & 0xFFFFFFFF) != (cs & 0xFFFFFFFF):  # SUM32 digest mismatch
            detected.append(p)
            radius += 255 * n * weight
        else:
            undetected_bytes += int((v != c).sum())
            silent = silent or vs != cs
    if silent:
        outcome = "silent_wrong"
    elif not detected:
        outcome = "undetected_corruption_answer_exact" if undetected_bytes else "vote_recovered"
    else:
        outcome = "bounded_degraded" if abs(err) <= radius else "cert_bound_failure"
    return {"outcome": outcome, "detected_planes": "|".join(map(str, detected)) or "none"}


# ---------------------------------------------------------------- driver

def run_gpu(bench: Path, ds: str, pdir: Path, r_vector: list[int], plan_file: Path, fam: str, seed: int,
            n: int, csv_path: Path) -> dict[str, str]:
    cmd = [str(bench), "--planes-dir", str(pdir), "--dataset", ds, "--r-vector", "|".join(map(str, r_vector)),
           "--fault-plan", str(plan_file), "--fault-family", fam, "--seed", str(seed), "--n-rows", str(n),
           "--iters", "10", "--csv", str(csv_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr.strip()[-400:])
        return {"outcome": "hard_fail", "latency_ms": "NA"}
    with csv_path.open() as f:
        return list(csv.DictReader(f))[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", action="append", required=True, metavar="NAME=DIR",
                    help="dataset name and directory with plane_NNN.bin files (repeatable)")
    ap.add_argument("--gpu-bench", type=Path)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--n-rows", type=int, default=500_000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if not args.cpu_only and (args.gpu_bench is None or not args.gpu_bench.exists()):
        sys.exit("--gpu-bench missing (or pass --cpu-only)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpu_csv = args.out_dir / "gpu_raw.csv"
    gpu_csv.write_text("")
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        for spec in args.planes:
            ds, pdir = spec.split("=", 1)
            pdir = Path(pdir)
            clean = load_planes(pdir, args.n_rows)
            lsb = max(available_planes(pdir))
            for policy, r_vector in POLICIES.items():
                for sc in FAULT_SCENARIOS:
                    for seed in SEEDS:
                        plan = fault_plan(policy, sc, seed, r_vector, args.n_rows, lsb)
                        cpu = cpu_oracle(clean, r_vector, plan)
                        row = {"dataset": ds, "n_rows": args.n_rows, "policy": policy,
                               "r_vector": "|".join(map(str, r_vector)), "fault_family": sc["fault_family"],
                               "seed": seed, "fault_entries": len(plan), "cpu_outcome": cpu["outcome"],
                               "cpu_detected_planes": cpu["detected_planes"]}
                        if not args.cpu_only:
                            pf = Path(tmp) / "plan.txt"
                            pf.write_text("".join(f"{p} {r} {o} {m}\n" for p, r, o, m in plan))
                            g = run_gpu(args.gpu_bench, ds, pdir, r_vector, pf, sc["fault_family"], seed,
                                        args.n_rows, gpu_csv)
                            row.update(gpu_outcome=g["outcome"], latency_ms=g.get("latency_ms", "NA"),
                                       match=str(g["outcome"] == cpu["outcome"]).lower())
                        rows.append(row)
                        print(f"{ds:<16s} {policy:<34s} {sc['fault_family']:<34s} s={seed} "
                              f"cpu={cpu['outcome']:<17s} gpu={row.get('gpu_outcome', '-')}")

    out = args.out_dir / "matrix.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} configurations, CPU outcomes: {dict(Counter(r['cpu_outcome'] for r in rows))}")
    if not args.cpu_only:
        print(f"GPU outcomes: {dict(Counter(r['gpu_outcome'] for r in rows))}")
        print(f"CPU/GPU match: {sum(r['match'] == 'true' for r in rows)}/{len(rows)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
