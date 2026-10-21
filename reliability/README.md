# Fail-soft byte planes under injected memory faults

This directory is the reliability half of the project. It asks whether the byte-plane layout used in the performance experiments can also make an analytical query fail soft when the data it reads is corrupted: detect the corruption, repair it where replicas allow, and otherwise return an interval that is guaranteed to contain the true answer instead of a silently wrong number.

This line of work was stopped. The mechanism behaves as designed on the tested fault matrices, but I did not find a workload where silent corruption is frequent enough to justify the cost. The limitations section below says what the numbers do and do not show; read it before quoting anything from here.

## Problem

A SUM over a column held in GPU memory has three possible outcomes when some bytes are corrupted: the job aborts, the answer is silently wrong, or the system notices and degrades. For an analytical query a silently wrong answer is the dangerous one. The byte-plane layout splits every value by significance (plane 0 holds the most significant byte of every row), so corruption in a low plane moves the answer by a small, bounded amount while corruption in plane 0 can move it by a lot. The idea is to spend redundancy on the high-significance planes and use cheap per-plane digests to tell which planes can no longer be trusted.

## Fault model and assumptions

- Faults are injected in software: an XOR mask is applied to chosen bytes of a plane buffer (or of one or more of its replicas) before the query reads it. This is not a hardware fault and not a model of ECC behaviour. Kernels, registers, control flow and the digests themselves are assumed fault-free.
- Reference digests are assumed intact. The harness computes them from an uncorrupted copy; a real system would compute them at write time and store them separately.
- Fault "domains" are logical labels. The naive and striping policies below differ only in whether one domain fault reaches all replicas of a plane or only one; no physical HBM channel or stack placement was controlled or measured.
- Fault rates are chosen to exercise the mechanism. They are not calibrated to any observed field corruption rate.

Fault families, single-replica experiment (R1, `scripts/single_replica_detect_bound.py`):

- `plane_uniform`: every plane gets `max(1, round(n * rate))` random byte positions XORed with a random mask in 1..255; rates 1e-6, 1e-5, 1e-4.
- `same_fault_all`: one row, the same mask in all planes (correlated control), rate label 1e-5.

Fault families, structured experiment (R2, `scripts/run_structured_faults.py`):

- `single_domain_burst`: 4 contiguous bytes of plane 0, mask 0xFF, landing on the replicas of one fault domain.
- `regional_multi_byte`: 64 contiguous bytes of plane 0, mask 0xFF, one domain.
- `column_like_repeated_offset`: 8 bytes of plane 0 at stride 4096, mask 0xFF, one domain.
- `plane_localized_msb`: 8 contiguous bytes of plane 0 in one replica, mask 0xFF.
- `plane_localized_lsb`: 8 contiguous bytes of the lowest plane in one replica, mask 0xFF.
- `same_fault_all_correlated_control`: 4 contiguous bytes of plane 0 in every replica, mask 0xFF.
- `mixed_independent_plus_correlated`: 4 bytes of plane 0 in every replica (0xFF) plus 4 bytes of the lowest plane in one replica (0x55).

Fault plans are deterministic functions of (policy, family, seed); seeds 0, 1, 2.

## Method

Layout and answer. A column of `n` rows is stored as up to 8 planes of `n` bytes, plane `p` holding byte `p` of a big-endian 64-bit code. The query answer is the SUM of the codes, `sum_p S_p * 2^(8*(7-p))` where `S_p` is the byte sum of plane `p`. This is a SUM in code space, a stand-in for the FP64 SUM computed by the performance kernels.

Replication. Each plane `p` has `r_p` replicas. A byte-level vote produces the delivered plane: `r = 3` majority (if all three differ, the largest value), `r = 2` the larger of the two bytes (two copies cannot out-vote each other, so a disagreement is left to the digest), `r = 1` pass-through.

Policies (replicas for planes 0..7, extra storage for an 8-plane column):

- `naive_same_region_replication_r3`: 3,1,1,1,1,1,1,1 (+25%); a domain fault hits all three copies.
- `logical_striping_diversity_r3`: 3,1,1,1,1,1,1,1 (+25%); a domain fault hits one copy.
- `graded_nmr_B11`: 3,2,1,1,1,1,1,1 (+37.5%).
- `uniformish_B11`: 2,1,2,1,2,1,1,1 (+37.5%), same budget spread over planes.
- `certified_degradation_fallback`: no replication (+0%), detection and interval only.

Detection. After the vote, a SUM32 digest of each plane is compared with the reference digest. The GPU path and the R2 oracle use the sum of bytes modulo 2^32; the R1 evaluator uses the sum of little-endian 32-bit words modulo 2^32.

Interval. For every plane whose digest mismatches, every one of its `n` bytes may be off by up to 255, so the delivered answer is reported as `delivered +/- radius`, `radius = sum over detected planes of 255 * n * 2^(8*(7-p))`.

Outcomes:

- `vote_recovered` / `recovered`: no digest mismatch and the answer is exact.
- `bounded_degraded` / `bounded`: mismatch detected, interval contains the clean answer.
- `cert_bound_failure` / `unavailable`: mismatch detected, interval misses the clean answer.
- `silent_wrong`: no mismatch but the answer differs from the clean answer.
- `undetected_corruption_answer_exact` (current code only): bytes corrupted, no mismatch, answer still exact.

## Injection method

- R1 runs entirely on the CPU (`scripts/single_replica_detect_bound.py`), single replica, on the full fields: `hurricane_u` (25,000,000 rows, 8 planes) and `cesm_atm_cloud` (168,480,000 rows, 7 planes; the missing eighth plane is treated as zeros). 2 datasets x (3 rates + 1 correlated control) x 3 seeds = 24 cells.
- R2 runs on an NVIDIA H200. For each configuration the driver writes the fault plan, evaluates it with a CPU oracle, launches the GPU binary once on the same planes and plan, and compares the two outcomes. Only the first 500,000 rows of each field are used. 2 datasets x 5 policies x 7 families x 3 seeds = 210 configurations.
- Inputs are byte planes of two SDRBench fields (Hurricane Isabel `U`, CESM-ATM `CLOUD`), produced by a fixed-point encoder with 4096-row segments. Neither the data nor the encoder is in this repository (the same limitation as Exp3 encoded mode).

## Results

R1, single replica (`results/r1_single_replica/`):

- 24 of 24 cells detected, 24 of 24 intervals contain the clean answer, 0 silent-wrong, 0 recovered. Every cell had a digest mismatch on every plane.
- Because every plane is faulted and nothing is replicated, the radius is 2.70x the clean answer for `hurricane_u` and 6.80x for `cesm_atm_cloud`, so the interval is wider than the answer itself and carries no useful information. The actual error was at most 3.4e-5 (`hurricane_u`) and 2.4e-4 (`cesm_atm_cloud`) of the answer.

R2, structured faults on the H200 (`results/r2_structured_faults/`):

- 210 configurations. GPU outcome equals CPU oracle outcome in 210 of 210 (`r2_cpu_gpu_oracle_comparison.csv`).
- 57 `vote_recovered`, 153 `bounded_degraded`, 0 `cert_bound_failure`, 0 `silent_wrong`, 0 hard failures (`r2_matrix.csv`, column `verdict_cell`).
- Recovered / bounded per policy, 42 configurations each: striping 24 / 18, graded 24 / 18, naive 6 / 36, uniformish 3 / 39, fallback 0 / 42. With three copies of plane 0 in separate domains, every single-copy plane-0 fault is repaired; with all copies in one domain (naive) only the single-replica family is. Two copies (uniformish) mostly only detect; its 3 recoveries are cases where the larger byte happened to be the clean one.
- No policy recovers `same_fault_all_correlated_control` or `mixed_independent_plus_correlated` (`r2_fault_family_summary.csv`): when every copy of a plane is hit the same way, replication does not help and only the interval is left.
- Interval tightness (`bound_width_norm` = radius / clean answer): the 15 `hurricane_u` cells where only the lowest plane was detected have radius about 4e-17 of the answer. Every bounded cell with plane 0 detected has radius at least 2.69x the answer, i.e. no useful information.
- In the 500,000-row slice of `cesm_atm_cloud` the clean answer is 0 (every byte is zero; `bound_width_norm` equals `bound_width` in all its bounded rows). Half of the R2 matrix therefore injects faults into all-zero data.

Combined: 24 + 210 = 234 injected-fault configurations, 0 silent-wrong outcomes. 24 of them come from the CPU evaluator, 210 from the H200 path checked against the CPU oracle. Of the 234, 57 were repaired by voting and 177 were detected and returned as an interval. See the first two limitations for what this count does not show.

`python3 scripts/summarize_results.py` recomputes every number in this section from the CSVs.

## Overhead

- Storage: +25% (one plane triplicated) to +37.5% (graded and uniformish), plus one 4-byte digest per plane.
- Runtime: not listed. The runtime overhead measurement compared the protected path against a raw FP64 baseline that has the same problem as the excluded filter experiments (see the main README); the baseline needs to be fixed before a ratio means anything. In that run the unfused digest-and-vote path was slower than the unprotected byte-plane scan; fusing vote and digest into the scan kernel was never implemented.
- The `latency_ms` column in `r2_gpu_raw.csv` (about 0.04 to 0.05 ms) times 16 unfused kernel launches over 500,000 rows. At that size launch overhead dominates, so it is not an overhead measurement and is not quoted.

## Limitations and negative findings

- Under the workloads and fault rates I measured, silent corruption that reaches the query is too rare, so the cost of this protection is not clearly worth paying. The project stopped for that reason, not because the mechanism failed. A stronger version of this work would start from a workload where wrong answers from memory corruption are actually observed and someone is harmed by them.
- Zero silent-wrong is close to guaranteed by construction for this query. With a byte-sum digest, a corruption that keeps a plane's byte sum also keeps that plane's contribution to the SUM, so a wrong answer can only slip through if the byte-sum change is a nonzero multiple of 2^32, which needs millions of corrupted bytes in one plane. The experiments validate the implementation (vote, digest, interval, GPU agreeing with the oracle); they are not evidence of robustness against an adversarial or worst-case fault. A different query (for example MIN or a filter) or a different digest would not have this property. The R1 word-sum digest has a small blind spot of its own that the random masks never hit.
- When plane 0 cannot be repaired, the interval is wider than the answer (R1 everywhere, most R2 bounded cells). "Contains the true answer" is then trivially true and says nothing. The interval is informative only when detection is confined to low planes, which is the case graded replication is meant to create.
- Software injection into data buffers only; no ECC, no hardware fault, no fault in compute. Placement diversity is logical.
- R2 uses a 500,000-row prefix, and the `cesm_atm_cloud` prefix is all zeros. One run per configuration, two fields, one GPU model.
- The fault rates and families are not calibrated to field data.
- R1 and R2 use different digest definitions (word sum vs byte sum).
- The code here is not byte-for-byte the code that produced the numbers. Differences: (1) the original R2 harness checked the interval at half the radius; any interval that contains the answer at half the radius also does at the full radius, so the reported outcomes stand. (2) The original GPU harness never computed `silent_wrong` (it wrote a constant). The zero for R2 rests on this argument: in every `vote_recovered` configuration either at most one of three copies was corrupted, so the majority vote restores the bytes exactly, or two copies were voted by taking the larger byte, so the byte sum can only grow and an unchanged digest implies unchanged bytes. The current code checks it directly with exact integer arithmetic and also reports undetected byte corruption. (3) Input is read from `plane_NNN.bin` files instead of the encoder's artifact format. (4) The fault plan is applied to host copies before upload instead of one device round-trip per entry; the resulting buffers are the same.
- The encoder and the two datasets are not included, so R1 and R2 cannot be reproduced from this repository alone. The smoke test below runs the same code on synthetic planes.
- This snapshot was built and smoke-tested on an RTX A4500 (sm_86), not re-run on an H200.

## Files

- `src/bench_failsoft_pipeline.cu`: GPU pipeline (replicas, vote, SUM32 digest, interval, classification), one configuration per run, appends one CSV row.
- `scripts/run_structured_faults.py`: R2 fault plans, CPU oracle, GPU sweep, CPU/GPU comparison (`--cpu-only` without a GPU).
- `scripts/single_replica_detect_bound.py`: R1 evaluator (CPU).
- `scripts/make_synthetic_planes.py`: synthetic 8-plane column for testing.
- `scripts/smoke_test.sh`: synthetic run of R1 and R2 plus one digest-blind-spot case.
- `scripts/summarize_results.py`: recomputes the numbers in this README.
- `results/r1_single_replica/`: per-cell CSVs per dataset and mode, and `r1_aggregate_summary.csv`.
- `results/r2_structured_faults/`: `r2_matrix.csv` (one row per configuration), `r2_cpu_gpu_oracle_comparison.csv`, `r2_gpu_raw.csv` (raw GPU rows), `r2_fault_plan_summary.csv`, `r2_policy_summary.csv`, `r2_fault_family_summary.csv`. These are the outputs of the original H200 run, column names unchanged; in `r2_matrix.csv` the `silent_wrong` column comes from the original harness (see limitations).

## Build and run

Requirements: CMake >= 3.24, CUDA toolkit, Python 3 with NumPy.

```bash
cmake -S reliability -B build/reliability -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/reliability -j
reliability/scripts/smoke_test.sh build/reliability/bench_failsoft_pipeline

# R2 on your own planes (directories with plane_000.bin .. plane_007.bin)
python3 reliability/scripts/run_structured_faults.py \
    --gpu-bench build/reliability/bench_failsoft_pipeline \
    --planes hurricane_u=<PLANES_DIR_1> --planes cesm_atm_cloud=<PLANES_DIR_2> \
    --n-rows 500000 --out-dir <OUT_DIR>

# R1
python3 reliability/scripts/single_replica_detect_bound.py --artifact-dir <PLANES_DIR_1> \
    --dataset hurricane_u --n-rows 25000000 --mode plane_uniform --output <OUT_DIR>/r1.csv

python3 reliability/scripts/summarize_results.py
```
