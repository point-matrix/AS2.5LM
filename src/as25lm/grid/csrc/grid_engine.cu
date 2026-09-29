// Grid Engine v2 -- CUDA implementation
// =====================================
// Mirrors grid_engine_v2/reference.py step for step. Summation order matches
// the reference (one thread walks each cell's points in stable-sorted order)
// and the extension is built with --fmad=false, so results are bit-identical.
//
// Orchestration uses ATen for sort / unique / select (CUB under the hood);
// the per-point and per-cell work is in the kernels below.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <optional>
#include <vector>

namespace {

constexpr int MAX_CLASSES = 32;
constexpr int MAX_LEVEL = 3;
constexpr int KEY_BITS = 22;
constexpr long long KEY_OFF = 1LL << (KEY_BITS - 1);
constexpr long long KEY_MASK = (1LL << KEY_BITS) - 1;
constexpr int LEVEL_SHIFT = 2 * KEY_BITS;
constexpr int N_OUT = 20;  // FINAL_DTYPE fields
constexpr int THREADS = 256;

// Index of each scalar in the params vector (see config.cuda_params()).
enum Param {
  P_FINEST = 0, P_TILE, P_EDGE3, P_EDGE2, P_EDGE1,
  P_SPLIT_PRI, P_SPLIT_MIN, P_MERGE_PRI,
  P_RANGE_LIMIT, P_STD_LIMIT, P_W_RANGE, P_W_STD, P_TRAV_PENALTY,
  P_NO_MERGE_R,
  P_COUNT
};

inline int blocks_for(int64_t n) { return static_cast<int>((n + THREADS - 1) / THREADS); }

__device__ __forceinline__ long long pack_key(long long level, long long ix, long long iy) {
  return (level << LEVEL_SHIFT) | ((ix + KEY_OFF) << KEY_BITS) | (iy + KEY_OFF);
}
__device__ __forceinline__ long long key_level(long long key) { return key >> LEVEL_SHIFT; }
__device__ __forceinline__ long long key_ix(long long key) { return ((key >> KEY_BITS) & KEY_MASK) - KEY_OFF; }
__device__ __forceinline__ long long key_iy(long long key) { return (key & KEY_MASK) - KEY_OFF; }

// ── Kernels ─────────────────────────────────────────────────────

// Finest-cell index, 40 cm tile, tile-centre band -> base-level cell key.
__global__ void k_point_keys(int64_t n, const float* __restrict__ xyz,
                             double finest, double tile,
                             double e3sq, double e2sq, double e1sq,
                             int64_t* __restrict__ i3x_out, int64_t* __restrict__ i3y_out,
                             int64_t* __restrict__ key_out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double x = (double)xyz[3 * i];
  const double y = (double)xyz[3 * i + 1];
  const long long lim = KEY_OFF - 1;
  long long ix = (long long)floor(x / finest);
  long long iy = (long long)floor(y / finest);
  ix = ix < -lim ? -lim : (ix > lim ? lim : ix);
  iy = iy < -lim ? -lim : (iy > lim ? lim : iy);

  const long long tx = ix >> MAX_LEVEL;
  const long long ty = iy >> MAX_LEVEL;
  const double cx = ((double)tx + 0.5) * tile;
  const double cy = ((double)ty + 0.5) * tile;
  const double r2 = cx * cx + cy * cy;

  long long level = 0;
  if (r2 < e1sq) level = 1;
  if (r2 < e2sq) level = 2;
  if (r2 < e3sq) level = 3;
  const int sh = MAX_LEVEL - (int)level;

  i3x_out[i] = ix;
  i3y_out[i] = iy;
  key_out[i] = pack_key(level, ix >> sh, iy >> sh);
}

// One thread per cell: walk its points (sorted, contiguous) and accumulate.
__global__ void k_reduce_cells(int64_t ncells,
                               const int64_t* __restrict__ offs, const int64_t* __restrict__ cnts,
                               const int64_t* __restrict__ perm,
                               const float* __restrict__ xyz, const int64_t* __restrict__ lab,
                               const float* __restrict__ conf, int nclasses,
                               double* __restrict__ o_sum, double* __restrict__ o_sumsq,
                               double* __restrict__ o_zmin, double* __restrict__ o_zmax,
                               int64_t* __restrict__ o_cls, double* __restrict__ o_wscore,
                               double* __restrict__ o_total) {
  int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (s >= ncells) return;
  double sc[MAX_CLASSES];
  for (int c = 0; c < nclasses; ++c) sc[c] = 0.0;
  double s1 = 0.0, s2 = 0.0, mn = INFINITY, mx = -INFINITY;
  const int64_t start = offs[s], end = offs[s] + cnts[s];
  for (int64_t j = start; j < end; ++j) {
    const int64_t p = perm[j];
    const double z = (double)xyz[3 * p + 2];
    s1 += z;
    s2 += z * z;
    mn = fmin(mn, z);
    mx = fmax(mx, z);
    sc[lab[p]] += (double)conf[p];
  }
  int w = 0;
  double best = sc[0];
  for (int c = 1; c < nclasses; ++c) {
    if (sc[c] > best) { best = sc[c]; w = c; }
  }
  double t = 0.0;
  for (int c = 0; c < nclasses; ++c) t += sc[c];
  o_sum[s] = s1;
  o_sumsq[s] = s2;
  o_zmin[s] = mn;
  o_zmax[s] = mx;
  o_cls[s] = w;
  o_wscore[s] = sc[w];
  o_total[s] = t;
}

__global__ void k_split_flags(int64_t ncells, const int64_t* __restrict__ key,
                              const int64_t* __restrict__ cnt, const int64_t* __restrict__ cls,
                              const int* __restrict__ pri_lut, int split_pri, long long split_min,
                              uint8_t* __restrict__ split) {
  int64_t s = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (s >= ncells) return;
  split[s] = (pri_lut[cls[s]] >= split_pri) && (key_level(key[s]) < MAX_LEVEL) &&
             (cnt[s] >= split_min);
}

// Points of split cells get their quadrant-child key; others keep theirs.
__global__ void k_rekey(int64_t n, const int64_t* __restrict__ ks, const int64_t* __restrict__ seg,
                        const uint8_t* __restrict__ split, const int64_t* __restrict__ perm,
                        const int64_t* __restrict__ i3x, const int64_t* __restrict__ i3y,
                        int64_t* __restrict__ key1) {
  int64_t j = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (j >= n) return;
  const long long k = ks[j];
  if (!split[seg[j]]) { key1[j] = k; return; }
  const long long level = key_level(k) + 1;
  const int sh = MAX_LEVEL - (int)level;
  const int64_t p = perm[j];
  key1[j] = pack_key(level, i3x[p] >> sh, i3y[p] >> sh);
}

__global__ void k_parent_keys(int64_t n, const int64_t* __restrict__ key, int64_t* __restrict__ parent) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long k = key[i];
  const long long level = key_level(k);
  parent[i] = level > 0 ? pack_key(level - 1, key_ix(k) >> 1, key_iy(k) >> 1) : -1;
}

// One thread per parent group. Members are child cells sorted by parent key;
// src < 0 marks a child that is not a leaf (it was split). Parents whose tile
// centre is within the no-merge radius never merge.
__global__ void k_merge_groups(int64_t ngroups,
                               const int64_t* __restrict__ gkeys,
                               const int64_t* __restrict__ goffs, const int64_t* __restrict__ gcnt,
                               const int64_t* __restrict__ msrc, double tile, double no_merge_r2,
                               const int64_t* __restrict__ c_cnt, const double* __restrict__ c_sum,
                               const double* __restrict__ c_sumsq, const double* __restrict__ c_zmin,
                               const double* __restrict__ c_zmax, const int64_t* __restrict__ c_cls,
                               const double* __restrict__ c_wscore, const double* __restrict__ c_total,
                               const int* __restrict__ pri_lut, int merge_pri,
                               uint8_t* __restrict__ elig,
                               int64_t* __restrict__ m_cnt, double* __restrict__ m_sum,
                               double* __restrict__ m_sumsq, double* __restrict__ m_zmin,
                               double* __restrict__ m_zmax, int64_t* __restrict__ m_cls,
                               double* __restrict__ m_wscore, double* __restrict__ m_total,
                               uint8_t* __restrict__ removed) {
  int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (g >= ngroups) return;
  const int64_t start = goffs[g], end = goffs[g] + gcnt[g];

  const long long pk = gkeys[g];
  const long long plevel = key_level(pk);
  const double cx = ((double)(key_ix(pk) >> plevel) + 0.5) * tile;
  const double cy = ((double)(key_iy(pk) >> plevel) + 0.5) * tile;
  bool ok = !(cx * cx + cy * cy < no_merge_r2);
  const int64_t first = msrc[start];
  if (first < 0) ok = false;
  const long long cls0 = ok ? c_cls[first] : -1;
  for (int64_t j = start; ok && j < end; ++j) {
    const int64_t s = msrc[j];
    if (s < 0 || c_cls[s] != cls0 || pri_lut[c_cls[s]] > merge_pri) ok = false;
  }
  elig[g] = ok ? 1 : 0;
  if (!ok) return;

  long long n = 0;
  double s1 = 0.0, s2 = 0.0, mn = INFINITY, mx = -INFINITY, ws = 0.0, tt = 0.0;
  for (int64_t j = start; j < end; ++j) {
    const int64_t s = msrc[j];
    n += c_cnt[s];
    s1 += c_sum[s];
    s2 += c_sumsq[s];
    mn = fmin(mn, c_zmin[s]);
    mx = fmax(mx, c_zmax[s]);
    ws += c_wscore[s];
    tt += c_total[s];
    removed[s] = 1;
  }
  m_cnt[g] = n;
  m_sum[g] = s1;
  m_sumsq[g] = s2;
  m_zmin[g] = mn;
  m_zmax[g] = mx;
  m_cls[g] = cls0;
  m_wscore[g] = ws;
  m_total[g] = tt;
}

// Final per-cell fields, columns in FINAL_DTYPE order.
__global__ void k_finalize(int64_t n, const int64_t* __restrict__ key, const int64_t* __restrict__ cnt,
                           const double* __restrict__ sum, const double* __restrict__ sumsq,
                           const double* __restrict__ zmin, const double* __restrict__ zmax,
                           const int64_t* __restrict__ cls, const double* __restrict__ wscore,
                           const double* __restrict__ total,
                           const int* __restrict__ pri_lut, const double* __restrict__ trav_lut,
                           double finest, double range_limit, double std_limit,
                           double w_range, double w_std, double trav_penalty,
                           double* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long k = key[i];
  const long long level = key_level(k);
  const long long ix = key_ix(k), iy = key_iy(k);
  const double size = finest * (double)(1LL << (MAX_LEVEL - level));
  const double x_min = (double)ix * size;
  const double x_max = (double)(ix + 1) * size;
  const double y_min = (double)iy * size;
  const double y_max = (double)(iy + 1) * size;

  const double c = (double)cnt[i];
  const double mean = sum[i] / c;
  const double var = fmax(sumsq[i] / c - mean * mean, 0.0);
  const double hr = zmax[i] - zmin[i];
  const double sd = sqrt(var);
  const double complexity = w_range * fmin(hr / range_limit, 1.0) + w_std * fmin(sd / std_limit, 1.0);
  const long long cl = cls[i];
  double trav = trav_lut[cl] * (1.0 - trav_penalty * complexity);
  trav = fmin(fmax(trav, 0.0), 1.0);
  const double conf = total[i] > 0.0 ? wscore[i] / total[i] : 0.0;

  double* o = out + i * N_OUT;
  o[0] = x_min;
  o[1] = x_max;
  o[2] = y_min;
  o[3] = y_max;
  o[4] = (x_min + x_max) * 0.5;
  o[5] = (y_min + y_max) * 0.5;
  o[6] = size;
  o[7] = (double)level;
  o[8] = mean;
  o[9] = zmin[i];
  o[10] = zmax[i];
  o[11] = var;
  o[12] = (double)cl;
  o[13] = conf;
  o[14] = 1.0;
  o[15] = complexity;
  o[16] = trav;
  o[17] = c;
  o[18] = (double)pri_lut[cl];
  o[19] = hr;
}

// Copy each output row into the numpy FINAL_DTYPE byte layout, so the host
// receives ready-made records instead of converting column by column.
// layout[0][k] = byte offset of field k, layout[1][k] = 1 for int32, 0 for
// float64. Offsets and itemsize are multiples of 4 (checked in Python), so
// every field is written as 4-byte words.
__global__ void k_pack(int64_t n, const double* __restrict__ mat, const int* __restrict__ layout,
                       int nfields, int itemsize, uint8_t* __restrict__ out) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double* row = mat + i * N_OUT;
  uint8_t* base = out + i * (int64_t)itemsize;
  for (int k = 0; k < nfields; ++k) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(base + layout[k]);
    if (layout[nfields + k]) {
      dst[0] = (uint32_t)(int32_t)row[k];
    } else {
      const unsigned long long bits = (unsigned long long)__double_as_longlong(row[k]);
      dst[0] = (uint32_t)(bits & 0xffffffffULL);
      dst[1] = (uint32_t)(bits >> 32);
    }
  }
}

// ── Host helpers ────────────────────────────────────────────────

struct Cells {
  at::Tensor key, cnt, sum, sumsq, zmin, zmax, cls, wscore, total;

  int64_t size() const { return key.size(0); }

  Cells select(const at::Tensor& idx) const {
    return {key.index_select(0, idx), cnt.index_select(0, idx), sum.index_select(0, idx),
            sumsq.index_select(0, idx), zmin.index_select(0, idx), zmax.index_select(0, idx),
            cls.index_select(0, idx), wscore.index_select(0, idx), total.index_select(0, idx)};
  }

  static Cells cat(const Cells& a, const Cells& b) {
    return {at::cat({a.key, b.key}), at::cat({a.cnt, b.cnt}), at::cat({a.sum, b.sum}),
            at::cat({a.sumsq, b.sumsq}), at::cat({a.zmin, b.zmin}), at::cat({a.zmax, b.zmax}),
            at::cat({a.cls, b.cls}), at::cat({a.wscore, b.wscore}), at::cat({a.total, b.total})};
  }

  static Cells empty(int64_t n, const at::TensorOptions& lopt, const at::TensorOptions& dopt) {
    return {at::empty({n}, lopt), at::empty({n}, lopt), at::empty({n}, dopt), at::empty({n}, dopt),
            at::empty({n}, dopt), at::empty({n}, dopt), at::empty({n}, lopt), at::empty({n}, dopt),
            at::empty({n}, dopt)};
  }
};

inline at::Tensor flat_nonzero(const at::Tensor& mask) { return at::nonzero(mask).view({-1}); }

inline std::tuple<at::Tensor, at::Tensor> stable_sort(const at::Tensor& t) {
  return at::sort(t, std::optional<bool>(true), /*dim=*/0, /*descending=*/false);
}

// Group points already sorted by key into cells. seg_out gets, for each
// sorted point, the index of its cell.
Cells reduce_sorted(const at::Tensor& ks, const at::Tensor& perm, const at::Tensor& xyz,
                    const at::Tensor& lab, const at::Tensor& conf, int nclasses,
                    cudaStream_t stream, at::Tensor* seg_out) {
  auto uc = at::unique_consecutive(ks, /*return_inverse=*/true, /*return_counts=*/true, std::nullopt);
  at::Tensor ukeys = std::get<0>(uc);
  at::Tensor counts = std::get<2>(uc);
  if (seg_out) *seg_out = std::get<1>(uc);
  at::Tensor offs = at::cumsum(counts, 0) - counts;
  const int64_t nc = ukeys.size(0);

  Cells c = Cells::empty(nc, ks.options(), ks.options().dtype(at::kDouble));
  c.key = ukeys;
  c.cnt = counts;
  if (nc > 0) {
    k_reduce_cells<<<blocks_for(nc), THREADS, 0, stream>>>(
        nc, offs.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), perm.data_ptr<int64_t>(),
        xyz.data_ptr<float>(), lab.data_ptr<int64_t>(), conf.data_ptr<float>(), nclasses,
        c.sum.data_ptr<double>(), c.sumsq.data_ptr<double>(), c.zmin.data_ptr<double>(),
        c.zmax.data_ptr<double>(), c.cls.data_ptr<int64_t>(), c.wscore.data_ptr<double>(),
        c.total.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return c;
}

at::Tensor parent_keys(const at::Tensor& key, cudaStream_t stream) {
  at::Tensor parent = at::empty_like(key);
  const int64_t n = key.size(0);
  if (n > 0) {
    k_parent_keys<<<blocks_for(n), THREADS, 0, stream>>>(n, key.data_ptr<int64_t>(),
                                                          parent.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return parent;
}

// Optional per-stage GPU timing with CUDA events.
struct StageTimer {
  bool on;
  cudaStream_t stream;
  std::vector<cudaEvent_t> events;

  StageTimer(bool on_, cudaStream_t s) : on(on_), stream(s) { mark(); }
  ~StageTimer() {
    for (auto e : events) cudaEventDestroy(e);
  }
  void mark() {
    if (!on) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, stream);
    events.push_back(e);
  }
  at::Tensor result() {
    const int64_t n = events.size() > 1 ? (int64_t)events.size() - 1 : 0;
    at::Tensor t = at::zeros({n}, at::kFloat);
    if (!on || n == 0) return t;
    cudaEventSynchronize(events.back());
    auto acc = t.accessor<float, 1>();
    for (int64_t i = 0; i < n; ++i) {
      float ms = 0.f;
      cudaEventElapsedTime(&ms, events[i], events[i + 1]);
      acc[i] = ms;
    }
    return t;
  }
};

}  // namespace

// Returns {matrix (n_cells x 20, float64, CUDA), xy_bounds (4, float32, CUDA),
//          stats (7, int64, CPU), stage_ms (5, float32, CPU),
//          packed (n_cells * pack_itemsize bytes, uint8, CUDA; empty if not requested)}.
// stats: n_points_in, n_points_used, n_stage1_cells, n_splits, n_merges,
//        n_final_cells, n_unlabeled_dropped
// stage_ms: filter+keys, cells, split, merge, finalize (zeros unless profile)
// pack_layout: (2, 20) int32 CUDA tensor of field offsets / int flags, or empty.
std::vector<at::Tensor> grid_engine_run(at::Tensor xyz_in, at::Tensor labels_in, at::Tensor conf_in,
                                        at::Tensor pri_lut_in, at::Tensor trav_lut_in,
                                        std::vector<double> P, bool profile,
                                        at::Tensor pack_layout, int64_t pack_itemsize) {
  TORCH_CHECK(xyz_in.is_cuda() && labels_in.is_cuda() && conf_in.is_cuda(),
              "xyz, labels and conf must be CUDA tensors");
  TORCH_CHECK(xyz_in.dim() == 2 && xyz_in.size(1) >= 3, "xyz must be (N, >=3)");
  TORCH_CHECK(labels_in.size(0) == xyz_in.size(0) && conf_in.size(0) == xyz_in.size(0),
              "xyz, labels and conf must have the same length");
  TORCH_CHECK((int64_t)P.size() == P_COUNT, "expected ", (int)P_COUNT, " params, got ", P.size());
  const int nclasses = (int)pri_lut_in.size(0);
  TORCH_CHECK(nclasses <= MAX_CLASSES && trav_lut_in.size(0) == nclasses, "bad class lookup tables");
  const bool pack = pack_layout.numel() > 0;
  if (pack) {
    TORCH_CHECK(pack_layout.is_cuda() && pack_layout.scalar_type() == at::kInt &&
                    pack_layout.numel() == 2 * N_OUT && pack_layout.is_contiguous(),
                "pack_layout must be a contiguous (2, 20) int32 CUDA tensor");
    TORCH_CHECK(pack_itemsize > 0 && pack_itemsize % 4 == 0, "pack_itemsize must be a positive multiple of 4");
  }

  c10::cuda::CUDAGuard guard(xyz_in.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  StageTimer timer(profile, stream);

  const auto dev = xyz_in.device();
  at::Tensor pri_lut = pri_lut_in.to(dev, at::kInt).contiguous();
  at::Tensor trav_lut = trav_lut_in.to(dev, at::kDouble).contiguous();

  // 1. Drop unlabeled points, compute point keys.
  const int64_t n_in = xyz_in.size(0);
  at::Tensor lab_all = labels_in.to(at::kLong);
  at::Tensor keep = flat_nonzero(lab_all.ne(0));
  at::Tensor xyz = xyz_in.slice(1, 0, 3).index_select(0, keep).to(at::kFloat).contiguous();
  at::Tensor lab = lab_all.index_select(0, keep).contiguous();
  at::Tensor conf = conf_in.index_select(0, keep).to(at::kFloat).contiguous();
  const int64_t n = xyz.size(0);

  at::Tensor stats = at::zeros({7}, at::kLong);
  auto st = stats.accessor<int64_t, 1>();
  st[0] = n_in;
  st[1] = n;
  st[6] = n_in - n;

  const auto lopt = lab.options();
  const auto dopt = lopt.dtype(at::kDouble);
  if (n == 0) {
    timer.mark();
    return {at::empty({0, N_OUT}, dopt), at::zeros({4}, xyz.options()), stats, timer.result(),
            at::empty({0}, lopt.dtype(at::kByte))};
  }
  TORCH_CHECK(lab.max().item<int64_t>() < nclasses && lab.min().item<int64_t>() >= 0,
              "labels must be in 0..", nclasses - 1);

  at::Tensor xy = xyz.slice(1, 0, 2);
  at::Tensor bounds = at::cat({at::amin(xy, at::IntArrayRef{0}), at::amax(xy, at::IntArrayRef{0})});  // xmin, ymin, xmax, ymax

  at::Tensor i3x = at::empty({n}, lopt), i3y = at::empty({n}, lopt), key0 = at::empty({n}, lopt);
  k_point_keys<<<blocks_for(n), THREADS, 0, stream>>>(
      n, xyz.data_ptr<float>(), P[P_FINEST], P[P_TILE],
      P[P_EDGE3] * P[P_EDGE3], P[P_EDGE2] * P[P_EDGE2], P[P_EDGE1] * P[P_EDGE1],
      i3x.data_ptr<int64_t>(), i3y.data_ptr<int64_t>(), key0.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  timer.mark();

  // 2. Stage-1 cells at the base level.
  auto s1 = stable_sort(key0);
  at::Tensor ks = std::get<0>(s1);
  at::Tensor perm = std::get<1>(s1);
  at::Tensor seg1;
  Cells cells1 = reduce_sorted(ks, perm, xyz, lab, conf, nclasses, stream, &seg1);
  const int64_t n1 = cells1.size();
  st[2] = n1;
  timer.mark();

  // 3. Split high-priority cells into quadrant children.
  at::Tensor split = at::empty({n1}, lopt.dtype(at::kByte));
  k_split_flags<<<blocks_for(n1), THREADS, 0, stream>>>(
      n1, cells1.key.data_ptr<int64_t>(), cells1.cnt.data_ptr<int64_t>(),
      cells1.cls.data_ptr<int64_t>(), pri_lut.data_ptr<int>(), (int)P[P_SPLIT_PRI],
      (long long)P[P_SPLIT_MIN], split.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  at::Tensor split_idx = flat_nonzero(split);
  const int64_t n_splits = split_idx.size(0);
  st[3] = n_splits;

  Cells cells2 = cells1;
  if (n_splits > 0) {
    at::Tensor key1 = at::empty_like(ks);
    k_rekey<<<blocks_for(n), THREADS, 0, stream>>>(
        n, ks.data_ptr<int64_t>(), seg1.data_ptr<int64_t>(), split.data_ptr<uint8_t>(),
        perm.data_ptr<int64_t>(), i3x.data_ptr<int64_t>(), i3y.data_ptr<int64_t>(),
        key1.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto s2 = stable_sort(key1);
    at::Tensor ks2 = std::get<0>(s2);
    at::Tensor perm2 = perm.index_select(0, std::get<1>(s2));
    cells2 = reduce_sorted(ks2, perm2, xyz, lab, conf, nclasses, stream, nullptr);
  }
  const int64_t n2 = cells2.size();
  timer.mark();

  // 4. Merge (single pass): parents whose children are all low-priority
  //    leaves of one class become a single leaf.
  at::Tensor par2 = parent_keys(cells2.key, stream);
  at::Tensor leaf_m = flat_nonzero(par2.ge(0));
  at::Tensor mkey = par2.index_select(0, leaf_m);
  at::Tensor msrc = leaf_m;
  if (n_splits > 0) {
    at::Tensor par_s = parent_keys(cells1.key.index_select(0, split_idx), stream);
    par_s = par_s.index_select(0, flat_nonzero(par_s.ge(0)));
    mkey = at::cat({mkey, par_s});
    msrc = at::cat({msrc, at::full({par_s.size(0)}, -1, msrc.options())});
  }

  at::Tensor removed = at::zeros({n2}, lopt.dtype(at::kByte));
  Cells merged;
  bool have_merged = false;
  int64_t n_merges = 0;
  if (mkey.size(0) > 0) {
    auto ms = stable_sort(mkey);
    at::Tensor mks = std::get<0>(ms);
    at::Tensor msrc_s = msrc.index_select(0, std::get<1>(ms)).contiguous();
    auto ug = at::unique_consecutive(mks, false, true, std::nullopt);
    at::Tensor gkeys = std::get<0>(ug);
    at::Tensor gcnt = std::get<2>(ug);
    at::Tensor goffs = at::cumsum(gcnt, 0) - gcnt;
    const int64_t ng = gkeys.size(0);

    at::Tensor elig = at::empty({ng}, lopt.dtype(at::kByte));
    Cells m = Cells::empty(ng, lopt, dopt);
    k_merge_groups<<<blocks_for(ng), THREADS, 0, stream>>>(
        ng, gkeys.data_ptr<int64_t>(), goffs.data_ptr<int64_t>(), gcnt.data_ptr<int64_t>(),
        msrc_s.data_ptr<int64_t>(), P[P_TILE], P[P_NO_MERGE_R] * P[P_NO_MERGE_R],
        cells2.cnt.data_ptr<int64_t>(), cells2.sum.data_ptr<double>(),
        cells2.sumsq.data_ptr<double>(), cells2.zmin.data_ptr<double>(),
        cells2.zmax.data_ptr<double>(), cells2.cls.data_ptr<int64_t>(),
        cells2.wscore.data_ptr<double>(), cells2.total.data_ptr<double>(),
        pri_lut.data_ptr<int>(), (int)P[P_MERGE_PRI], elig.data_ptr<uint8_t>(),
        m.cnt.data_ptr<int64_t>(), m.sum.data_ptr<double>(), m.sumsq.data_ptr<double>(),
        m.zmin.data_ptr<double>(), m.zmax.data_ptr<double>(), m.cls.data_ptr<int64_t>(),
        m.wscore.data_ptr<double>(), m.total.data_ptr<double>(), removed.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    m.key = gkeys;
    at::Tensor elig_idx = flat_nonzero(elig);
    n_merges = elig_idx.size(0);
    if (n_merges > 0) {
      merged = m.select(elig_idx);
      have_merged = true;
    }
  }
  st[4] = n_merges;

  Cells fin = cells2.select(flat_nonzero(removed.eq(0)));
  if (have_merged) fin = Cells::cat(fin, merged);
  fin = fin.select(std::get<1>(stable_sort(fin.key)));
  const int64_t nf = fin.size();
  st[5] = nf;
  timer.mark();

  // 5. Final per-cell fields.
  at::Tensor out = at::empty({nf, N_OUT}, dopt);
  if (nf > 0) {
    k_finalize<<<blocks_for(nf), THREADS, 0, stream>>>(
        nf, fin.key.data_ptr<int64_t>(), fin.cnt.data_ptr<int64_t>(), fin.sum.data_ptr<double>(),
        fin.sumsq.data_ptr<double>(), fin.zmin.data_ptr<double>(), fin.zmax.data_ptr<double>(),
        fin.cls.data_ptr<int64_t>(), fin.wscore.data_ptr<double>(), fin.total.data_ptr<double>(),
        pri_lut.data_ptr<int>(), trav_lut.data_ptr<double>(), P[P_FINEST], P[P_RANGE_LIMIT],
        P[P_STD_LIMIT], P[P_W_RANGE], P[P_W_STD], P[P_TRAV_PENALTY], out.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // 6. Optional: the same rows in the numpy FINAL_DTYPE byte layout.
  at::Tensor packed = at::empty({pack ? nf * pack_itemsize : 0}, lopt.dtype(at::kByte));
  if (pack && nf > 0) {
    k_pack<<<blocks_for(nf), THREADS, 0, stream>>>(nf, out.data_ptr<double>(), pack_layout.data_ptr<int>(),
                                                   N_OUT, (int)pack_itemsize, packed.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  timer.mark();

  return {out, bounds, stats, timer.result(), packed};
}
