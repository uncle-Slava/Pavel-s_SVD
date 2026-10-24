// Template: your SVD with your own CUDA kernel (real float32/float64 matrices, GPU only).
//
//     python -m svdlab check examples/native/my_svd.cu --device cuda --compare torch
//
// One-sided Jacobi: each round of a sweep rotates n/2 disjoint column pairs; one thread
// block per pair computes <w_p,w_p>, <w_q,w_q>, <w_p,w_q> and applies the rotation to the
// columns p, q of W and V. The host loops over rounds and sweeps.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

using torch::Tensor;
constexpr int THREADS = 256;

template <typename T>
__global__ void rotate_pairs(T* w, T* v, int64_t m, int64_t n, const int64_t* P, const int64_t* Q, T tol,
                             T* correlation) {
  const int64_t p = P[blockIdx.x], q = Q[blockIdx.x];
  __shared__ T sa[THREADS], sb[THREADS], sg[THREADS];
  __shared__ T c, s;
  T a = 0, b = 0, g = 0;
  for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {  // W is row-major (m, n)
    T x = w[i * n + p], y = w[i * n + q];
    a += x * x; b += y * y; g += x * y;
  }
  sa[threadIdx.x] = a; sb[threadIdx.x] = b; sg[threadIdx.x] = g;
  __syncthreads();
  for (int half = blockDim.x / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) {
      sa[threadIdx.x] += sa[threadIdx.x + half];
      sb[threadIdx.x] += sb[threadIdx.x + half];
      sg[threadIdx.x] += sg[threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    T alpha = sa[0], beta = sb[0], gamma = sg[0];
    T denom = sqrt(alpha * beta);
    T corr = denom > 0 ? fabs(gamma) / denom : T(0);
    correlation[blockIdx.x] = corr;
    if (corr > tol) {
      T zeta = (beta - alpha) / (2 * gamma);
      T t = (zeta >= 0 ? T(1) : T(-1)) / (fabs(zeta) + sqrt(1 + zeta * zeta));
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    } else {
      c = 1; s = 0;
    }
  }
  __syncthreads();
  if (s == 0) return;
  for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
    T x = w[i * n + p], y = w[i * n + q];
    w[i * n + p] = c * x - s * y;
    w[i * n + q] = s * x + c * y;
  }
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    T x = v[i * n + p], y = v[i * n + q];
    v[i * n + p] = c * x - s * y;
    v[i * n + q] = s * x + c * y;
  }
}

std::tuple<Tensor, Tensor, Tensor> svd(Tensor a, int64_t rank) {
  TORCH_CHECK(a.is_cuda(), "this example runs on the GPU: use --device cuda");
  TORCH_CHECK(a.dim() == 2 && !a.is_complex(), "this example supports real float32/float64 matrices");
  const bool wide = a.size(0) < a.size(1);
  Tensor w = (wide ? a.t() : a).contiguous().clone();
  const int64_t m = w.size(0), n = w.size(1);
  Tensor v = torch::eye(n, w.options());

  // Round-robin schedule: pairs[r] = disjoint (p, q) for round r, padded with an unused column.
  std::vector<int64_t> players(n + n % 2);
  std::iota(players.begin(), players.end(), 0);
  std::vector<std::pair<Tensor, Tensor>> rounds;
  for (size_t r = 0; r + 1 < players.size(); ++r) {
    std::vector<int64_t> p, q;
    for (size_t i = 0; i < players.size() / 2; ++i) {
      int64_t x = players[i], y = players[players.size() - 1 - i];
      if (std::max(x, y) < n) { p.push_back(std::min(x, y)); q.push_back(std::max(x, y)); }
    }
    rounds.emplace_back(torch::tensor(p).to(w.device()), torch::tensor(q).to(w.device()));
    std::rotate(players.begin() + 1, players.end() - 1, players.end());
  }
  Tensor correlation = torch::zeros({(int64_t)rounds.size(), std::max<int64_t>(n / 2, 1)}, w.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "jacobi_svd", [&] {
    const scalar_t tol = 10 * std::numeric_limits<scalar_t>::epsilon();
    for (int sweep = 0; sweep < 60; ++sweep) {
      correlation.zero_();
      for (size_t r = 0; r < rounds.size(); ++r) {
        const auto& [p, q] = rounds[r];
        if (p.numel() == 0) continue;
        rotate_pairs<scalar_t><<<p.numel(), THREADS, 0, stream>>>(
            w.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), m, n, p.data_ptr<int64_t>(), q.data_ptr<int64_t>(),
            tol, correlation[r].data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      if (correlation.max().item<double>() <= tol) break;
    }
  });

  Tensor sigma = w.square().sum(0).sqrt();
  Tensor order = sigma.argsort(0, /*descending=*/true);
  sigma = sigma.index_select(0, order);
  Tensor u = w.index_select(1, order) / torch::where(sigma > 0, sigma, torch::ones_like(sigma));
  v = v.index_select(1, order);
  const int64_t k = std::min(rank, n);
  u = u.narrow(1, 0, k).contiguous();
  sigma = sigma.narrow(0, 0, k);
  Tensor vh = v.narrow(1, 0, k).t().contiguous();
  if (wide) return {vh.t().contiguous(), sigma, u.t().contiguous()};
  return {u, sigma, vh};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("svd", &svd, "one-sided Jacobi SVD with a CUDA kernel"); }
