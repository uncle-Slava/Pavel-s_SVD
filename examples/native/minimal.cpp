// The smallest native method: checks that your C++ toolchain works, and shows how to export
// a building block next to the SVD.
//
//     python -m svdlab check examples/native/minimal.cpp --quick
//
// svd() calls the library SVD (LAPACK on CPU tensors, cuSOLVER on GPU tensors), so every case
// should be ok. qr() has the contract of svdlab.blocks.qr and can be plugged into a recipe:
//
//     from svdlab import native, recipes
//     module = native.build('examples/native/minimal.cpp')
//     recipes.randomized(a, rank, qr=module.qr)
#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(torch::Tensor a, int64_t rank) {
  auto [u, s, vh] = torch::linalg_svd(a, /*full_matrices=*/false);
  const int64_t k = std::min<int64_t>(rank, s.size(0));
  return {u.narrow(1, 0, k), s.narrow(0, 0, k), vh.narrow(0, 0, k)};
}

// Tall a (m, n) -> Q (m, n) with orthonormal columns, R (n, n) upper triangular.
std::tuple<torch::Tensor, torch::Tensor> qr(torch::Tensor a) {
  TORCH_CHECK(a.dim() == 2 && a.size(0) >= a.size(1), "qr expects one tall matrix");
  return torch::linalg_qr(a, "reduced");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("svd", &svd, "library SVD truncated to rank");
  m.def("qr", &qr, "library QR, same contract as svdlab.blocks.qr");
}
