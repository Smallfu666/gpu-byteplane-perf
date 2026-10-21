#!/usr/bin/env python3
"""Single-replica detect-and-bound evaluator (CPU, experiment R1).

No replication (r = 1 for every plane). For each cell:

  load clean planes -> inject faults -> per-plane SUM32 digest -> detect
  -> certified interval -> check that the interval contains the clean answer

Outcomes:
  recovered:     no digest mismatch and the delivered answer equals the clean one
  bounded:       digest mismatch, interval contains the clean answer
  unavailable:   digest mismatch, interval does not contain the clean answer
  silent_wrong:  no digest mismatch but the delivered answer differs

The digest here is the sum of little-endian 32-bit words of the plane, modulo
2^32. (The GPU path in src/ uses the sum of bytes modulo 2^32.)

Example:
  python3 scripts/single_replica_detect_bound.py --artifact-dir <PLANES_DIR> \
      --dataset hurricane_u --n-rows 25000000 --rates 1e-6 1e-5 1e-4 \
      --seeds 0 1 2 --mode plane_uniform --output r1_hurricane_u_plane_uniform.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PLANE_COUNT = 8
PLANE_WEIGHTS = [1 << (8 * (7 - p)) for p in range(PLANE_COUNT)]


def sum32(data: bytes) -> int:
    arr = np.frombuffer(data, dtype=np.uint8)
    pad = (4 - arr.size % 4) % 4
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    return int(arr.view(dtype=np.uint32).sum()) & 0xFFFFFFFF


def load_planes(artifact_dir: Path, n_rows: int) -> list[bytes]:
    planes: list[bytes] = []
    for p in range(PLANE_COUNT):
        path = artifact_dir / f"plane_{p:03d}.bin"
        data = path.read_bytes()[:n_rows] if path.is_file() else bytes(n_rows)
        if len(data) < n_rows:
            data = data + bytes(n_rows - len(data))
        planes.append(data)
    return planes


def inject_plane_uniform(clean: list[bytes], n_rows: int, rate: float, rng: random.Random) -> list[bytearray]:
    """Every plane gets max(1, round(n_rows * rate)) random byte positions XORed with a random nonzero mask."""
    delivered = [bytearray(p) for p in clean]
    for p in range(PLANE_COUNT):
        n_faults = max(1, int(n_rows * rate + 0.5)) if rate > 0 else 0
        if n_faults > 0:
            for pos in rng.sample(range(n_rows), min(n_faults, n_rows)):
                delivered[p][pos] ^= rng.randint(1, 255)
    return delivered


def inject_same_fault_all(clean: list[bytes], n_rows: int, rng: random.Random) -> list[bytearray]:
    """One row position, same mask, in every plane (correlated control)."""
    delivered = [bytearray(p) for p in clean]
    pos = rng.randint(0, n_rows - 1)
    mask = rng.randint(1, 255)
    for p in range(PLANE_COUNT):
        delivered[p][pos] ^= mask
    return delivered


def encoded_sum(planes: list[bytes]) -> int:
    return sum(sum(p) * PLANE_WEIGHTS[i] for i, p in enumerate(planes))


def bound_radius(detected: list[bool], n_rows: int) -> float:
    """Worst-case |delivered - clean|: every byte of every detected plane off by 255.

    The certified interval is delivered +/- radius. It is conditional on every
    corrupted plane being detected; an undetected corrupted plane is a
    silent_wrong outcome.
    """
    return sum(255.0 * PLANE_WEIGHTS[p] * n_rows for p in range(PLANE_COUNT) if detected[p])


def run_cell(clean: list[bytes], delivered: list[bytearray], clean_digests: list[int], n_rows: int,
             fault_planes: list[int], fault_offsets: list[int]) -> dict[str, Any]:
    detected_planes = [sum32(bytes(d)) != c for d, c in zip(delivered, clean_digests)]
    detected = any(detected_planes)
    clean_answer = encoded_sum(clean)
    delivered_answer = encoded_sum(delivered)

    if not detected:
        outcome = "recovered" if delivered_answer == clean_answer else "silent_wrong"
        width = 0.0
        contains_truth = delivered_answer == clean_answer
    else:
        width = bound_radius(detected_planes, n_rows)
        contains_truth = abs(delivered_answer - clean_answer) <= width
        outcome = "bounded" if contains_truth else "unavailable"

    return {
        "detected": detected,
        "detected_planes": str([p for p in range(PLANE_COUNT) if detected_planes[p]]),
        "contains_truth": contains_truth,
        "outcome": outcome,
        "bound_width": width,
        "clean_answer": float(clean_answer),
        "delivered_answer": float(delivered_answer),
        "fault_planes": str(sorted(set(fault_planes))),
        "fault_offsets": len(fault_offsets),
    }


def evaluate(artifact_dir: Path, dataset: str, n_rows: int, rates: list[float], seeds: list[int],
             mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean = load_planes(artifact_dir, n_rows)
    clean_digests = [sum32(p) for p in clean]
    for rate in rates:
        for seed in seeds:
            rng = random.Random(seed)
            if mode == "plane_uniform":
                delivered = inject_plane_uniform(clean, n_rows, rate, rng)
                fault_planes = list(range(PLANE_COUNT))
                # Kept from the original runs so the RNG stream (and the
                # fault_offsets column) matches the reported CSVs.
                fault_offsets: list[int] = []
                for _ in range(PLANE_COUNT):
                    nf = max(1, int(n_rows * rate + 0.5)) if rate > 0 else 0
                    if nf > 0:
                        fault_offsets.extend(rng.sample(range(n_rows), min(nf, n_rows)))
            elif mode == "same_fault_all":
                delivered = inject_same_fault_all(clean, n_rows, rng)
                fault_planes = list(range(PLANE_COUNT))
                fault_offsets = [rng.randint(0, n_rows - 1)]
            else:
                raise ValueError(f"unknown mode: {mode}")
            r = run_cell(clean, delivered, clean_digests, n_rows, fault_planes, fault_offsets)
            r.update(dataset=dataset, n_rows=n_rows, mode=mode, rate=str(rate), seed=seed)
            rows.append(r)
            print(f"  rate={rate:.1e} seed={seed} {r['outcome']:<13s} radius={r['bound_width']:.3e}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-dir", type=Path, required=True, help="directory with plane_000.bin .. plane_007.bin")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-rows", type=int, required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mode", default="plane_uniform", choices=["plane_uniform", "same_fault_all"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    rows = evaluate(args.artifact_dir, args.dataset, args.n_rows, args.rates, args.seeds, args.mode)

    fields = ["dataset", "n_rows", "mode", "rate", "seed", "detected", "detected_planes", "contains_truth",
              "outcome", "bound_width", "clean_answer", "delivered_answer", "fault_planes", "fault_offsets"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    counts = Counter(r["outcome"] for r in rows)
    print(f"\n{len(rows)} cells: {dict(counts)}  ({time.perf_counter() - t0:.1f} s)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
