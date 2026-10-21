#!/usr/bin/env bash
# Smoke test on synthetic planes: R1 evaluator, R2 matrix (GPU vs CPU oracle),
# and one digest-blind-spot case. Usage: scripts/smoke_test.sh <bench_failsoft_pipeline> [n_rows]
set -euo pipefail

BENCH=${1:?path to bench_failsoft_pipeline}
N=${2:-200000}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

python3 "$HERE/make_synthetic_planes.py" --out-dir "$OUT/planes" --n-rows "$N"
python3 "$HERE/single_replica_detect_bound.py" --artifact-dir "$OUT/planes" --dataset synthetic \
    --n-rows "$N" --output "$OUT/r1.csv" | tail -2
python3 "$HERE/run_structured_faults.py" --gpu-bench "$BENCH" --planes "synthetic=$OUT/planes" \
    --n-rows "$N" --out-dir "$OUT/r2" | tail -4
python3 - "$OUT/r2/matrix.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r for r in rows if r["match"] != "true" or r["cpu_outcome"] in ("silent_wrong", "cert_bound_failure")]
sys.exit(f"FAIL: {len(bad)} rows mismatch or fail" if bad else 0)
EOF

# +1 on one byte and -1 on another byte of plane 7, single replica: the SUM32
# digest and the SUM answer are both unchanged, so the harness must report the
# corruption as undetected (not as recovered).
python3 - "$OUT/planes/plane_007.bin" "$OUT/cancel.txt" <<'EOF'
import sys
p = open(sys.argv[1], "rb").read()
i = next(k for k in range(len(p)) if p[k] < 255)
j = next(k for k in range(i + 1, len(p)) if p[k] > 0)
open(sys.argv[2], "w").write(f"7 0 {i} {p[i] ^ (p[i] + 1)}\n7 0 {j} {p[j] ^ (p[j] - 1)}\n")
EOF
"$BENCH" --planes-dir "$OUT/planes" --dataset synthetic --r-vector "1|1|1|1|1|1|1|1" \
    --fault-plan "$OUT/cancel.txt" --fault-family digest_blind_spot --n-rows "$N" --csv "$OUT/blind.csv" 2>/dev/null
grep -q ",undetected_corruption_answer_exact$" "$OUT/blind.csv" || { echo "FAIL: blind-spot case"; exit 1; }
echo "smoke test passed"
