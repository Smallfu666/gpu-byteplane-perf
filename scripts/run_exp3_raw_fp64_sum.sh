#!/usr/bin/env bash
# Generic Slurm template for the raw FP64 SUM comparator (Exp3).
# Fill in the placeholders for your own cluster, or ignore the #SBATCH lines
# and run this script directly on a machine with an sm_90 GPU.
#SBATCH --job-name=exp3_raw_fp64_sum
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --chdir=<WORKDIR>

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Directory holding <dataset>.f64le.bin raw little-endian FP64 files.
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory with raw FP64 inputs}"
RUN_DIR="$ROOT_DIR/results/exp3_raw_fp64_sum_baseline/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# Load your own CUDA toolchain (nvcc, CMake) before running this script.
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "GPU: $gpu_name"

BUILD_DIR="$ROOT_DIR/build/exp3"
cmake -S "$ROOT_DIR/benchmarks/experiment3" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 >/dev/null
cmake --build "$BUILD_DIR" --target bench_raw_fp64_sum -j >/dev/null
BIN="$BUILD_DIR/bench_raw_fp64_sum"

datasets=(heavy_tailed sensor uniform zipfian)
per_dataset_csvs=()
for ds in "${datasets[@]}"; do
  out_csv="$RUN_DIR/raw_fp64_sum_${ds}.csv"
  "$BIN" --device 0 --input "$DATA_ROOT/${ds}.f64le.bin" --dataset "$ds" \
    --block 256 --grid_mul 1 --warmup 10 --iters 200 --validate --csv "$out_csv"
  per_dataset_csvs+=("$out_csv")
done

final_csv="$RUN_DIR/raw_fp64_sum_baseline.csv"
python3 - "$final_csv" "${per_dataset_csvs[@]}" <<'PY'
import csv, sys
from pathlib import Path

final_csv = Path(sys.argv[1])
keep = ["dataset", "aggregation", "baseline_type", "n", "logical_bytes", "logical_GBps",
        "estimated_physical_GBps", "rows_per_sec", "cpu_sum", "gpu_sum", "abs_error",
        "rel_error", "validated", "device", "kernel_path"]
rows = []
for p in sys.argv[2:]:
    with open(p, newline="") as f:
        row = next(csv.DictReader(f))
    rows.append({k: row[k] for k in keep})
with final_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keep)
    w.writeheader()
    w.writerows(rows)
PY
echo "final CSV: $final_csv"
