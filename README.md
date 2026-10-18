# GPU Byte-plane Scan & Performance Optimization on NVIDIA H200

CUDA kernels and benchmarks for scanning a column stored as byte-planes, with Nsight Compute (NCU) evidence for what limits each kernel on an NVIDIA H200.

## Overview

A 64-bit float column is stored as separate byte-planes: plane 0 holds the most significant byte of every row, plane 1 the next byte, and so on. A query that tolerates an approximate answer reads only the first `k` planes, so it moves roughly `k/8` of the bytes of a plain FP64 column. The engineering question here is: when a kernel reads fewer bytes, does it take proportionally less time, and what stops it if not?

- **Exp0**: memory-access microbenchmark (coalesced, half-masked and random-gather reads).
- **Exp1 (headline)**: byte-plane scan kernels from 8-bit to 128-bit loads, with NCU and SASS attribution.
- **Exp3 (second implementation story)**: a progressive SUM kernel, and a specialization of its first real-data version that removed runtime-indexed local-memory accesses.

This is not a database engine and makes no comparison against any named library. Each bandwidth figure below is labelled logical (bytes the algorithm needed, divided by time) or NCU-measured (DRAM counters). "Logical GB/s" is not HBM bandwidth.

## Scope

Included: Exp0, Exp1, Exp3 sources, build files, run scripts and small result summaries (CSV and text).

Not included: filter and filter+aggregate experiments. They are excluded pending a re-measurement against a fair (bandwidth-saturating) raw baseline, so no numbers from them appear here. Profiler binaries (`.ncu-rep`) are also not included; only the extracted CSV summaries are.

## Environment

All numbers: one NVIDIA H200 (132 SMs, HBM3e), CUDA 12.6, sm_90, compiled with `-O3 --use_fast_math -lineinfo` (Exp3 adds `--ptxas-options=-v`). Runs were done on a Slurm-managed H200 node. Each configuration was run once; no variance or confidence intervals are available.

## Data layout

- Structure of arrays: `plane[p][row]`, one `uint8_t` per row per plane, each plane contiguous.
- `rowpack16`: one thread loads 16 consecutive rows of the same plane as one `uint4` (`LDG.E.128`), so a warp requests 512 contiguous bytes of one plane per instruction. `k` is the number of planes read and is unrelated to the row packing.
- Exp3 (encoded mode) cuts a column into segments (4096 rows in the reported runs), each with a base value and per-plane scale; planes after the first `k` are not read.

## Exp1: byte-plane scan, 8-bit to 128-bit loads (headline)

Files: `benchmarks/experiment1/exp1_kernels_byte.cuh` (`scan_planes_u8_byte_unrolled<K>`, `scan_planes_u8_byte_ilp4<K>`), `exp1_kernels_rowpack.cuh` (`scan_planes_u8_rowpack4<K>`, `scan_planes_u8_rowpack16<K>`), driver `bench_byteplane_scan.cu`.

The workload is deliberately trivial: sum all bytes of the first `k` planes of 1e8 rows, so the kernel is close to a pure scan. Variants:

- scalar baseline: grid-stride loop, one `LDG.E.U8` per plane per row, `K` template-unrolled.
- ILP4: same loads, four independent rows in flight per thread.
- rowpack4: `uint32` load, 4 rows per load, byte-sum via shifts and masks.
- rowpack16: `uint4` load, 16 rows per load.
- shared128 (diagnostic): stage 128 B per warp in shared memory; 4x overfetch, kept as a negative control.

Setup: n = 1e8, block 256, grid 1056 (8 blocks per SM), 10 warmup and 1000 timed iterations, CUDA events, single run per cell, no L2 flush between iterations. At k = 1 the working set is about 100 MB, so part of it can be L2-resident.

- Time per iteration, k = 1: scalar 0.140 ms, ILP4 0.063 ms, rowpack4 0.044 ms, rowpack16 0.027 ms.
- Time per iteration, k = 8: scalar 0.380 ms, ILP4 0.334 ms, rowpack4 0.184 ms, rowpack16 0.182 ms.
- rowpack16 vs scalar: 5.1x at k = 1, 3.3x at k = 2, 2.1x at k = 4 and k = 8.
- rowpack16 vs ILP4: 2.3x at k = 1, 1.8x at k = 4 and k = 8.
- rowpack16 vs rowpack4: 1.6x at k = 1, 1.05x at k = 4, 1.01x at k = 8. Past a few planes 32-bit loads are already close to enough; 128-bit loads matter most when the kernel reads little data.

NCU attribution (k = 8 unless noted; NCU kernel durations are not the same as the CUDA-event times above; one profiled launch per kernel):

- DRAM throughput as percent of peak: ILP4 46.8%, rowpack4 80.8%, rowpack16 89.7%.
- Executed instructions: ILP4 121.0M, rowpack4 74.9M, rowpack16 56.7M. At k = 1: 36.7M, 13.9M, 8.4M.
- SASS load opcodes confirm the width change: `LDG.E.U8` (ILP4), `LDG.E` 32-bit (rowpack4), `LDG.E.128` (rowpack16).
- At k = 8, rowpack16 reads about 4.3 TB/s per NCU, close to the benchmark's logical figure (4.39 TB/s). They agree here because the planes are contiguous and every fetched byte is used.

ILP4 is a partial negative result: it helps at k = 1 and 2 (about 2.2x over scalar at k = 1 in these runs) and is nearly a no-op from k = 5 up, consistent with the kernel being limited by the number of load instructions rather than loads in flight.

Data: `results/exp1/run_20260415_rowpack_full_after_impl/*.csv`, `results/exp1/ncu_rowpack_after_impl_20260416/` (`ncu_summary.csv`, `sass_load_summary.csv`, per-run bench CSVs).

## Exp3: progressive SUM and a local-memory specialization

Files: `benchmarks/experiment3/exp3_kernels_progressive.cuh` (`progressive_sum_rowpack16_u8subcols_runtime`, `progressive_sum_real_rowpack16_specialized<K_READ, PLANE_COUNT>`), raw baseline `bench_raw_fp64_sum.cu`, driver `bench_progressive_aggregation.cu`.

Design: one `unsigned long long` accumulator per plane per thread (integer sum of that plane's bytes), one block reduction per plane, and a final SUM combined in FP64 as `n*base + sum_p plane_sum[p] * basis[p]`.

The first real-data (encoded) kernel kept per-plane accumulators in a runtime-indexed array (`plane_sums[32]`). Profiling notes from the original development (not included here) showed local-memory use (`cuobjdump` stack frame and `LDL`/`STL` instructions in the hot path) that the synthetic template kernel did not have. I specialized the kernel: `K_READ` and `PLANE_COUNT` became template parameters, per-plane accumulation is selected at compile time (`if constexpr`), runtime indexing is gone, and the loops and control flow were restructured and unrolled accordingly. These changes were made together, so the measured difference is not attributable to any single one of them.

Result: execution time 0.832 ms to 0.290 ms (2.87x) on the tested H200 workload, i.e. 1.20 to 3.44 TB/s logical, with identical GPU sums between the two kernels. Single dataset (uniform, n = 1e8, refinement depth 9, 10 subcolumns), single run, 1000 iterations, no variance, no L2 flush. Data: `results/exp3_real_specialized_ab/`.

Precision-throughput curve for the specialized kernel, n = 1e8, 200 iterations, single run per point, CUDA events, logical bytes only. Raw baseline is a vectorized raw FP64 SUM (`raw_fp64_sum_vec2`): 0.186 ms for 800 MB, 4.31 TB/s logical.

- k = 1: 0.048 ms, 3.9x over raw
- k = 2: 0.068 ms, 2.7x
- k = 3: 0.086 ms, 2.2x
- k = 4: 0.104 ms, 1.8x
- k = 5: 0.125 ms, 1.5x
- k = 6: 0.166 ms, 1.1x
- k = 7: 0.181 ms, 1.03x
- k = 8: 0.219 ms, 0.85x (slower than raw)

The curve mixes datasets: k = 1 to 6 are read from uniform-distribution artifacts, k = 7 and 8 from heavy-tailed and zipfian artifacts (rows for k = 7 and 8 exist only for those distributions in the retained summary). Timings across distributions and artifact precisions are within about 1% of each other where they overlap. The ideal `8/k` is not reached (3.9x at k = 1, not 8x), and throughput is non-monotonic (k = 5 reaches 3.99 TB/s logical, k = 6 drops to 3.61). I have not profiled why. Data: `results/exp3_v2_full_specialized_sum/exp3_v2_full_sum_throughput_all.csv`, `results/exp3_raw_fp64_sum_baseline/`.

## Exp0: memory-access baseline

File: `benchmarks/experiment0/bench_hbm_bw.cu` (`seq_read_dummy_reduce`, `masked_read_dummy_reduce`, `gather_read_dummy_reduce`), `uint4` loads, 1 MiB to 8 GiB, 200 iterations, single run, no L2 flush.

- Sequential coalesced read at 8 GiB: 4.48 TB/s (logical).
- Sequential at 32 MiB: 5.36 TB/s logical. That exceeds HBM peak and reflects L2 residency; it is not HBM bandwidth. Read the sweep at sizes well above L2.
- Every other lane active: 2.24 TB/s reported, because only active lanes' bytes are counted. The kernel time (1.917 ms) equals the sequential time at the same span (1.916 ms), consistent with whole 32-byte sectors being fetched; I have not verified the physical traffic with NCU.
- Random gather of 16-byte elements over an 8 GiB span: 0.59 TB/s logical, about 13% of sequential.

These give the reference used above: a scan of this kind can at best approach roughly 4.4 TB/s of delivered read throughput on this GPU, and Exp1 rowpack16 at k = 8 reaches it.

## Summary of claims

Supported by the evidence in this repo:

- Moving a byte-plane scan from 8-bit to 128-bit loads raises NCU DRAM throughput from about 47% to 90% and cuts executed instructions by about 2.1x at k = 8 (4.4x at k = 1).
- Specializing the Exp3 progressive SUM kernel to eliminate runtime-indexed local-memory accesses reduced execution time 0.832 ms to 0.290 ms on the tested workload (single dataset, single run).
- Progressive SUM is faster than a vectorized raw FP64 SUM for k <= 7 on this setup (1.03x at k = 7) and slower at k = 8.

Not claimed: a speedup over any named library, any end-to-end or filter/aggregate result, or that logical GB/s equals HBM bandwidth.

## Reproduction

Honest status: the builds and runs behind these numbers were done on an H200 node under Slurm. This repository snapshot was assembled afterwards on a machine without `nvcc` and has not been rebuilt or re-run in its current form.

Requirements taken from the build files and run scripts: CMake >= 3.24, C++17 and CUDA 17 standard, CUDA 12.6 toolkit used for the reported runs, `-DCMAKE_CUDA_ARCHITECTURES=90` (sm_90 GPU; other GPUs will not match the numbers).

```bash
# Exp0
cmake -S benchmarks/experiment0 -B build/exp0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp0 -j
./build/exp0/bench_hbm_bw --mode seq --csv exp0_seq.csv

# Exp1 (no input data needed)
cmake -S benchmarks/experiment1 -B build/exp1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp1 -j
./build/exp1/bench_byteplane_scan --n 100000000 --strategy rowpack16 --k_min 1 --k_max 8 --warmup 10 --iters 1000 --validate --csv rowpack16.csv

# Exp3 (synthetic mode needs no input data)
cmake -S benchmarks/experiment3 -B build/exp3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build/exp3 -j
```

NCU as used for the Exp1 tables: `ncu --set full --target-processes all -k regex:scan_planes_u8_ -f -o out ./build/exp1/bench_byteplane_scan --n 100000000 --strategy rowpack16 --k_min 8 --k_max 8 --warmup 0 --iters 1`. `scripts/ncu_command_template.sh` and `results/exp1/ncu_rowpack_after_impl_20260416/extract_ncu_summary.py` show the capture and summary extraction.

Limits on reproducing the Exp3 numbers:

- The generic-versus-specialized A/B and the precision-throughput curve use `--mode encoded_dev_subcolumns`, which reads a pre-encoded byte-plane artifact (layout in `exp3_real_data_layout.hpp`). The encoder that generated those artifacts is not included in this repository, so this mode cannot be reproduced from this snapshot alone. Synthetic mode (`synthetic_fixed_point_subcolumns`) runs without external data.
- `scripts/run_exp3_raw_fp64_sum.sh` is a generic Slurm template with `<ACCOUNT>`, `<PARTITION>`, `<WORKDIR>` placeholders and needs raw little-endian FP64 inputs in `DATA_ROOT`.
- Paths in result CSVs that pointed at machine-specific locations were replaced by placeholders (`<DATA_ROOT>`, `<ENCODED_ROOT>`, `<RUN_DIR>`); job IDs were removed.
- `run_meta`-style files and profiler binaries were not copied. The Exp1 run `run_20260415_rowpack_full_after_impl` has no recorded run metadata, so its exact command lineage is incomplete; the configuration listed above is what the CSVs show.

## Limitations

- One GPU model, one CUDA version, single runs per cell, no confidence intervals.
- No L2 flush between iterations. At k = 1 in Exp1 (about 100 MB) part of the working set can be L2-resident; the NCU DRAM percentages are consistent with mostly-DRAM traffic, but a flush-based run would be cleaner.
- The contiguous 64-bit scan in Exp1 is a side reference only. It reached about 80% DRAM throughput under NCU, below rowpack16, so it is not the best possible raw scan. The Exp0 `uint4` sequential read (4.48 TB/s logical) is the better ceiling.
- The unexplained Exp3 behaviour (k = 6 dip, k = 8 slower than raw, and the k = 1 floor of 0.048 ms versus 0.027 ms for the Exp1 kernel on the same bytes) is not profiled here. Per-plane block reductions and register pressure are candidates, not findings.
- `--use_fast_math` is enabled for all CUDA targets. The integer paths are unaffected and the FP64 sums were checked against CPU references, but it is a flag worth questioning.
- Exp3 encoded-mode results depend on an artifact generator that is not part of this repository.
