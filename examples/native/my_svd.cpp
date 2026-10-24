// Template: your SVD in C++ (ATen tensor operations; the same code runs on CPU and GPU).
//
//     python -m svdlab check examples/native/my_svd.cpp --compare torch
//     python -m svdlab check examples/native/my_svd.cpp --device cuda
//
// svdlab compiles this file with torch.utils.cpp_extension (cached until the file changes)
// and calls svd(a, rank). Same algorithm as examples/my_svd.py: one-sided Jacobi.
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <numeric>
#include <tuple>
#include <vector>

using torch::Tensor;

// Rounds of disjoint column pairs (p, q) covering every pair once per sweep.
static std::vector<std::pair<Tensor, Tensor>> round_robin(int64_t n, torch::Device device) {
  std::vector<int64_t> players(n + n % 2);
  std::iota(players.begin(), players.end(), 0);
  std::vector<std::pair<Tensor, Tensor>> rounds;
  for (size_t r = 0; r + 1 < players.size(); ++r) {
    std::vector<int64_t> p, q;
    for (size_t i = 0; i < players.size() / 2; ++i) {
      int64_t x = players[i], y = players[players.size() - 1 - i];
      if (std::max(x, y) < n) {
        p.push_back(std::min(x, y));
        q.push_back(std::max(x, y));
      }
    }
    rounds.emplace_back(torch::tensor(p).to(device), torch::tensor(q).to(device));
    std::rotate(players.begin() + 1, players.end() - 1, players.end());
  }
  return rounds;
}

std::tuple<Tensor, Tensor, Tensor> svd(Tensor a, int64_t rank) {
  TORCH_CHECK(a.dim() == 2, "expected one matrix");
  const bool wide = a.size(0) < a.size(1);
  Tensor w = (wide ? a.transpose(0, 1).conj() : a).clone().resolve_conj();
  const int64_t n = w.size(1);
  Tensor v = torch::eye(n, w.options());
  const bool fp64 = c10::toRealValueType(w.scalar_type()) == torch::kDouble;
  const double tol = 10 * (fp64 ? DBL_EPSILON : FLT_EPSILON), tiny = fp64 ? DBL_MIN : FLT_MIN;
  const auto rounds = round_robin(n, w.device());

  for (int sweep = 0; sweep < 60; ++sweep) {
    Tensor off = torch::zeros({}, w.options().dtype(c10::toRealValueType(w.scalar_type())));
    for (const auto& [p, q] : rounds) {
      Tensor wp = w.index_select(1, p), wq = w.index_select(1, q);
      Tensor alpha = wp.abs().square().sum(0), beta = wq.abs().square().sum(0);
      Tensor gamma = (wp.conj() * wq).sum(0);
      Tensor g = gamma.abs();
      Tensor correlation = g / (alpha * beta).sqrt().clamp_min(tiny);
      off = torch::maximum(off, correlation.max());
      Tensor zeta = (beta - alpha) / (2 * g.clamp_min(tiny));
      Tensor t = torch::where(zeta == 0, torch::ones_like(zeta), zeta.sign() / (zeta.abs() + (1 + zeta.square()).sqrt()));
      Tensor c = (1 + t.square()).rsqrt();
      Tensor rotate = correlation > tol;
      Tensor s = torch::where(rotate, c * t, torch::zeros_like(c));
      c = torch::where(rotate, c, torch::ones_like(c));
      Tensor phase = torch::where(g > 0, gamma / g.clamp_min(tiny), torch::ones_like(gamma));
      for (Tensor* x : {&w, &v}) {  // same rotation on W and on V
        Tensor xp = x->index_select(1, p), xq = x->index_select(1, q);
        x->index_copy_(1, p, c * xp - s * phase.conj() * xq);
        x->index_copy_(1, q, s * phase * xp + c * xq);
      }
    }
    if (off.item<double>() <= tol) break;
  }

  Tensor sigma = w.abs().square().sum(0).sqrt();
  Tensor order = sigma.argsort(0, /*descending=*/true);
  sigma = sigma.index_select(0, order);
  w = w.index_select(1, order);
  v = v.index_select(1, order);
  Tensor u = w / torch::where(sigma > 0, sigma, torch::ones_like(sigma));
  const int64_t k = std::min(rank, n);
  u = u.narrow(1, 0, k);
  sigma = sigma.narrow(0, 0, k);
  Tensor vh = v.narrow(1, 0, k).transpose(0, 1).conj().resolve_conj();
  if (wide) return {vh.transpose(0, 1).conj().resolve_conj(), sigma, u.transpose(0, 1).conj().resolve_conj()};
  return {u.contiguous(), sigma, vh.contiguous()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("svd", &svd, "one-sided Jacobi SVD"); }
