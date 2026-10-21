// Fail-soft byte-plane pipeline on the GPU: per-plane replicas -> byte vote ->
// per-plane SUM32 digest -> detect -> certified interval -> classify.
//
// One process handles one configuration (dataset, replica vector, fault plan),
// so a driver can sweep configurations and compare each result against a CPU
// oracle (see scripts/run_structured_faults.py).
//
// Input: a directory with plane_000.bin .. plane_007.bin, one uint8 per row per
// plane, plane 0 = most significant byte. Missing planes are treated as absent.
//
// Fault plan: text file, one entry per line: "plane replica offset mask".
// Each entry XORs `mask` into byte `offset` of replica `replica` of `plane`
// before the pipeline runs (software injection into device buffers).
//
// Build: cmake -S reliability -B build/reliability -DCMAKE_CUDA_ARCHITECTURES=90
//        cmake --build build/reliability -j

#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kMaxPlanes = 8;
constexpr int kMaxReplicas = 3;

[[noreturn]] void die(const std::string &m) {
  std::fprintf(stderr, "FATAL: %s\n", m.c_str());
  std::exit(2);
}

void cuda_check(cudaError_t e, const char *w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", w, cudaGetErrorString(e));
    std::exit(2);
  }
}

double time_ms(const std::function<void(cudaStream_t)> &fn, int iters, cudaStream_t s) {
  for (int i = 0; i < 3; i++) fn(s);
  cuda_check(cudaStreamSynchronize(s), "warmup");
  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start), "event");
  cuda_check(cudaEventCreate(&stop), "event");
  cuda_check(cudaEventRecord(start, s), "record");
  for (int i = 0; i < iters; i++) fn(s);
  cuda_check(cudaEventRecord(stop, s), "record");
  cuda_check(cudaEventSynchronize(stop), "sync");
  float ms = 0;
  cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cuda_check(cudaEventDestroy(start), "event");
  cuda_check(cudaEventDestroy(stop), "event");
  return static_cast<double>(ms) / iters;
}

int parse_r_vector(const std::string &s, int *out) {
  int count = 0;
  size_t pos = 0;
  while (pos <= s.size() && count < kMaxPlanes) {
    size_t bar = s.find('|', pos);
    std::string tok = s.substr(pos, bar == std::string::npos ? std::string::npos : bar - pos);
    if (tok.empty()) break;
    out[count++] = std::stoi(tok);
    if (bar == std::string::npos) break;
    pos = bar + 1;
  }
  return count;
}

}  // namespace

// Byte-level vote. r=1: pass-through. r=2: max of the two copies (two copies
// cannot out-vote each other; a mismatch is left for the digest to detect).
// r=3: majority; if all three differ, the largest value wins (same tie rule
// as the CPU oracle).
__global__ void byte_vote(const uint8_t *__restrict__ r0, const uint8_t *__restrict__ r1,
                          const uint8_t *__restrict__ r2, uint8_t *__restrict__ voted,
                          uint64_t n, int r) {
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint8_t a = r0[i];
    if (r == 1) {
      voted[i] = a;
    } else if (r == 2) {
      const uint8_t b = r1[i];
      voted[i] = a >= b ? a : b;
    } else {
      const uint8_t b = r1[i], c = r2[i];
      uint8_t v;
      if (a == b || a == c) v = a;
      else if (b == c) v = b;
      else v = max(a, max(b, c));
      voted[i] = v;
    }
  }
}

// SUM32 digest: sum of all bytes of a plane modulo 2^32. Grid-stride loop,
// one partial per block, final reduction on the host.
__global__ void sum32_partial(const uint8_t *__restrict__ src, uint64_t n,
                              uint32_t *__restrict__ block_partials) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_count = blockDim.x >> 5;
  const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
  uint32_t local = 0;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + tid; i < n; i += stride)
    local += static_cast<uint32_t>(src[i]);
  for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xFFFFFFFFu, local, off);
  __shared__ uint32_t s_red[32];
  if ((tid & 31) == 0) s_red[tid >> 5] = local;
  __syncthreads();
  if (tid < 32) {
    uint32_t w = (tid < warp_count) ? s_red[tid] : 0;
    for (int off = 16; off > 0; off >>= 1) w += __shfl_down_sync(0xFFFFFFFFu, w, off);
    if (tid == 0) block_partials[blockIdx.x] = w;
  }
}

static uint32_t digest(const uint8_t *d_src, uint64_t n, uint32_t *d_partial, int grid, int block,
                       cudaStream_t st) {
  sum32_partial<<<grid, block, 0, st>>>(d_src, n, d_partial);
  cuda_check(cudaStreamSynchronize(st), "digest");
  std::vector<uint32_t> h(static_cast<size_t>(grid));
  cuda_check(cudaMemcpy(h.data(), d_partial, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
             "digest D2H");
  uint32_t total = 0;
  for (uint32_t v : h) total += v;
  return total;
}

struct FaultEntry {
  int plane, replica;
  uint64_t offset;
  uint8_t mask;
};

static std::vector<FaultEntry> load_fault_plan(const std::string &path) {
  std::vector<FaultEntry> plan;
  if (path.empty()) return plan;
  std::ifstream f(path);
  if (!f.is_open()) die("cannot open fault plan: " + path);
  int p, r;
  unsigned long long o;
  unsigned m;
  while (f >> p >> r >> o >> m) plan.push_back({p, r, o, static_cast<uint8_t>(m)});
  return plan;
}

static std::vector<std::vector<uint8_t>> load_planes(const std::string &dir, uint64_t &n) {
  std::vector<std::vector<uint8_t>> planes;
  for (int p = 0; p < kMaxPlanes; p++) {
    char name[32];
    std::snprintf(name, sizeof(name), "/plane_%03d.bin", p);
    std::ifstream f(dir + name, std::ios::binary);
    if (!f.is_open()) break;
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    planes.push_back(std::move(data));
  }
  if (planes.empty()) die("no plane_000.bin found in " + dir);
  uint64_t shortest = planes[0].size();
  for (const auto &pl : planes) shortest = std::min<uint64_t>(shortest, pl.size());
  if (n == 0 || n > shortest) n = shortest;
  return planes;
}

int main(int argc, char **argv) {
  std::string planes_dir, dataset = "dataset", csv_path = "failsoft_gpu.csv", plan_path;
  std::string fault_family = "none";
  int r_vector[kMaxPlanes] = {3, 2, 1, 1, 1, 1, 1, 1};
  int iters = 10, seed = 0;
  uint64_t n = 0;

  for (int i = 1; i < argc; ++i) {
    std::string_view a(argv[i]);
    auto val = [&]() -> std::string {
      if (++i >= argc) die("missing value");
      return argv[i];
    };
    if (a == "--help" || a == "-h") {
      std::fprintf(stderr,
                   "Usage: %s --planes-dir DIR [--dataset NAME] [--r-vector \"3|2|1|1|1|1|1|1\"]\n"
                   "  [--fault-plan PATH] [--fault-family LABEL] [--seed N] [--n-rows N]\n"
                   "  [--iters N] [--csv PATH]\n",
                   argv[0]);
      return 0;
    } else if (a == "--planes-dir") planes_dir = val();
    else if (a == "--dataset") dataset = val();
    else if (a == "--r-vector") parse_r_vector(val(), r_vector);
    else if (a == "--fault-plan") plan_path = val();
    else if (a == "--fault-family") fault_family = val();
    else if (a == "--seed") seed = std::stoi(val());
    else if (a == "--n-rows") n = std::strtoull(val().c_str(), nullptr, 10);
    else if (a == "--iters") iters = std::stoi(val());
    else if (a == "--csv") csv_path = val();
    else die("unknown argument: " + std::string(a));
  }
  if (planes_dir.empty()) die("--planes-dir required");

  auto host_planes = load_planes(planes_dir, n);
  const int planes = static_cast<int>(host_planes.size());

  cuda_check(cudaSetDevice(0), "device");
  cudaDeviceProp prop;
  cuda_check(cudaGetDeviceProperties(&prop, 0), "props");
  std::fprintf(stderr, "Device: %s (sm_%d%d)  dataset=%s n=%" PRIu64 " planes=%d\n", prop.name,
               prop.major, prop.minor, dataset.c_str(), n, planes);

  // Replicas per plane, with the fault plan applied on the host before upload.
  int rep[kMaxPlanes];
  std::vector<std::vector<std::vector<uint8_t>>> host_rep(planes);
  for (int p = 0; p < planes; p++) {
    rep[p] = std::max(1, std::min(r_vector[p], kMaxReplicas));
    host_rep[p].assign(rep[p], std::vector<uint8_t>(host_planes[p].begin(), host_planes[p].begin() + n));
  }
  const auto plan = load_fault_plan(plan_path);
  for (const auto &e : plan) {
    if (e.plane < 0 || e.plane >= planes || e.replica < 0 || e.replica >= rep[e.plane] || e.offset >= n)
      die("fault plan entry out of range");
    host_rep[e.plane][e.replica][e.offset] ^= e.mask;
  }

  const int block = 256;
  const uint64_t blocks_needed = (n + block - 1) / block;
  const int grid = static_cast<int>(std::min<uint64_t>(blocks_needed, 512));
  cudaStream_t st;
  cuda_check(cudaStreamCreate(&st), "stream");

  std::vector<uint8_t *> d_clean(planes), d_voted(planes);
  std::vector<std::vector<uint8_t *>> d_rep(planes);
  for (int p = 0; p < planes; p++) {
    cuda_check(cudaMalloc(&d_clean[p], n), "malloc");
    cuda_check(cudaMemcpy(d_clean[p], host_planes[p].data(), n, cudaMemcpyHostToDevice), "H2D");
    cuda_check(cudaMalloc(&d_voted[p], n), "malloc");
    d_rep[p].resize(rep[p]);
    for (int j = 0; j < rep[p]; j++) {
      cuda_check(cudaMalloc(&d_rep[p][j], n), "malloc");
      cuda_check(cudaMemcpy(d_rep[p][j], host_rep[p][j].data(), n, cudaMemcpyHostToDevice), "H2D");
    }
  }
  uint32_t *d_partial = nullptr;
  cuda_check(cudaMalloc(&d_partial, grid * sizeof(uint32_t)), "malloc");

  // Reference digests. In a real system these are computed once at write time
  // and stored; here they are computed from the uncorrupted copy.
  std::vector<uint32_t> ref_digest(planes);
  for (int p = 0; p < planes; p++) ref_digest[p] = digest(d_clean[p], n, d_partial, grid, block, st);

  // Timed region: vote + digest for every plane (unfused, one launch each).
  const double latency_ms = time_ms(
      [&](cudaStream_t s) {
        for (int p = 0; p < planes; p++) {
          const int r = rep[p];
          byte_vote<<<grid, block, 0, s>>>(d_rep[p][0], d_rep[p][r > 1 ? 1 : 0],
                                           d_rep[p][r > 2 ? 2 : 0], d_voted[p], n, r);
        }
        for (int p = 0; p < planes; p++) sum32_partial<<<grid, block, 0, s>>>(d_voted[p], n, d_partial);
      },
      iters, st);
  cuda_check(cudaGetLastError(), "kernels");

  // Detection and exact answer comparison. The answer is the SUM over rows of
  // the encoded integer value, i.e. sum_p (sum of plane p bytes) * 2^(8*(7-p)).
  // Exact 128-bit integer arithmetic is used so an error in a low plane is not
  // lost to FP64 rounding.
  std::vector<int> detected_planes;
  __int128 answer_error = 0;
  unsigned __int128 radius = 0;
  uint64_t undetected_byte_mismatches = 0;
  bool undetected_answer_diff = false;
  std::vector<uint8_t> h_voted(n);
  for (int p = 0; p < planes; p++) {
    const uint32_t d = digest(d_voted[p], n, d_partial, grid, block, st);
    cuda_check(cudaMemcpy(h_voted.data(), d_voted[p], n, cudaMemcpyDeviceToHost), "D2H voted");
    uint64_t clean_sum = 0, voted_sum = 0, mism = 0;
    for (uint64_t i = 0; i < n; i++) {
      clean_sum += host_planes[p][i];
      voted_sum += h_voted[i];
      mism += (h_voted[i] != host_planes[p][i]);
    }
    const int shift = 8 * (kMaxPlanes - 1 - p);
    const __int128 delta = static_cast<__int128>(voted_sum) - static_cast<__int128>(clean_sum);
    answer_error += delta * (static_cast<__int128>(1) << shift);
    if (d != ref_digest[p]) {
      detected_planes.push_back(p);
      radius += (static_cast<unsigned __int128>(255) * n) << shift;
    } else {
      undetected_byte_mismatches += mism;
      if (delta != 0) undetected_answer_diff = true;
    }
  }
  const bool detected = !detected_planes.empty();
  const unsigned __int128 abs_err =
      answer_error < 0 ? static_cast<unsigned __int128>(-answer_error) : static_cast<unsigned __int128>(answer_error);
  // Certified interval: delivered +/- radius, radius = sum over detected planes
  // of 255 * n * plane_weight (worst case: every byte of a detected plane off by 255).
  const bool contains_truth = abs_err <= radius;

  const char *outcome;
  // A SUM answer is unchanged by a corruption that preserves the byte sum of a
  // plane, which is exactly the SUM32 blind spot; such cells are reported
  // separately rather than as recovered.
  if (!detected && undetected_answer_diff) outcome = "silent_wrong";
  else if (!detected) outcome = undetected_byte_mismatches ? "undetected_corruption_answer_exact" : "vote_recovered";
  else if (undetected_answer_diff) outcome = "silent_wrong";  // a corrupted plane slipped past its digest
  else outcome = contains_truth ? "bounded_degraded" : "cert_bound_failure";

  double clean_answer = 0.0;
  for (int p = 0; p < planes; p++) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; i++) s += host_planes[p][i];
    clean_answer += static_cast<double>(s) * static_cast<double>(1ULL << (8 * (kMaxPlanes - 1 - p)));
  }
  const double radius_d = static_cast<double>(radius);
  const double radius_norm = clean_answer != 0.0 ? radius_d / clean_answer : -1.0;

  std::string r_str, det_str;
  for (int p = 0; p < kMaxPlanes; p++) r_str += (p ? "|" : "") + std::to_string(r_vector[p]);
  for (size_t i = 0; i < detected_planes.size(); i++)
    det_str += (i ? "|" : "") + std::to_string(detected_planes[i]);
  if (det_str.empty()) det_str = "none";

  FILE *f = std::fopen(csv_path.c_str(), "a");
  if (!f) die("cannot open CSV " + csv_path);
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fprintf(f,
                 "dataset,n_rows,r_vector,fault_family,seed,fault_entries,latency_ms,detected_planes,"
                 "undetected_byte_mismatches,bound_radius,bound_radius_norm,contains_truth,outcome\n");
  std::fprintf(f, "%s,%" PRIu64 ",%s,%s,%d,%zu,%.6f,%s,%" PRIu64 ",%.6e,%.6e,%s,%s\n", dataset.c_str(), n,
               r_str.c_str(), fault_family.c_str(), seed, plan.size(), latency_ms, det_str.c_str(),
               undetected_byte_mismatches, radius_d, radius_norm, contains_truth ? "true" : "false", outcome);
  std::fclose(f);
  std::fprintf(stderr, "outcome=%s detected_planes=%s latency_ms=%.6f\n", outcome, det_str.c_str(), latency_ms);

  for (int p = 0; p < planes; p++) {
    for (auto *b : d_rep[p]) cuda_check(cudaFree(b), "free");
    cuda_check(cudaFree(d_voted[p]), "free");
    cuda_check(cudaFree(d_clean[p]), "free");
  }
  cuda_check(cudaFree(d_partial), "free");
  cuda_check(cudaStreamDestroy(st), "stream");
  return 0;
}
