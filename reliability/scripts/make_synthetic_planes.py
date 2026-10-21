#!/usr/bin/env python3
"""Write a synthetic 8-plane byte-plane column for smoke tests.

Each row is a smooth signal plus noise, encoded as an unsigned 64-bit
fixed-point code and split big-endian into plane_000.bin (most significant
byte) .. plane_007.bin (least significant byte). The real-data runs reported
in the README used planes produced by an encoder that is not part of this
repository; this generator only exists so the harness can run without them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-rows", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    i = np.arange(args.n_rows, dtype=np.float64)
    x = 50.0 + 15.0 * np.sin(i / 5000.0) + rng.normal(0.0, 2.0, args.n_rows)
    x = np.clip(x, 0.0, 127.999)
    codes = (x / 128.0 * 2.0**63).astype(np.uint64)
    # FP64 keeps 53 significant bits, so the bottom bytes of `codes` are zero;
    # fill them with noise so every plane carries data.
    codes |= rng.integers(0, 1 << 16, size=args.n_rows, dtype=np.uint64)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    as_bytes = codes.astype(">u8").view(np.uint8).reshape(args.n_rows, 8)
    for p in range(8):
        (args.out_dir / f"plane_{p:03d}.bin").write_bytes(np.ascontiguousarray(as_bytes[:, p]).tobytes())
    print(f"wrote 8 planes x {args.n_rows} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
