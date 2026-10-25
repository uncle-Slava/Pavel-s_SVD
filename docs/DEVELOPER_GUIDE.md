# Developer guide: write, build, run and check an SVD

This page is the compact reference. For a step-by-step introduction start with the
[tutorials](tutorials/README.md) and the [notebooks](../notebooks/).

Everything here goes through one package, `svdlab`, and one command, `python -m svdlab`.
Install once from the repository root with `python -m pip install -e .` (choose the right
PyTorch build for your device first; do not install a CUDA build over a ROCm one).

## 1. The function contract

```python
def svd(a, rank):
    return U, s, Vh
```

- `a` is a `torch.Tensor` of shape `(m, n)`, float32/float64/complex64/complex128, already on
  the device you asked for (`--device`). Do not modify it in place.
- `rank` is how many singular triplets are needed. It equals `min(m, n)` for a full SVD and
  is smaller for the low-rank cases.
- Return `U (m, k)`, `s (k,)` descending and non-negative, `Vh (k, n)` with `k >= rank`.
  Extra triplets are cut to `rank`, so a full SVD is always a valid answer. Torch tensors
  or NumPy arrays both work. Returning only `s` checks the singular values alone.
- On the GPU return GPU tensors. A result on the host is reported in `output_device`, and
  `--gpu-check` fails it, because the timing would then not be of a GPU computation.

## 2. The command line

```text
python -m svdlab check METHOD [METHOD ...] [options]
python -m svdlab list
```

A METHOD is one of:

| Form | Example |
| --- | --- |
| Python file with a function `svd` (or `my_svd`) | `my_svd.py` |
| A named function in a file | `my_svd.py:fast_variant` |
| C++ or CUDA source file | `my_svd.cpp`, `my_svd.cu` |
| Importable `module:function` | `svdlab.recipes:golub_kahan` |
| Builtin name | `torch`, `numpy`, `jacobi`, `bidiag_qr`, `polar_qdwh`, ... (`python -m svdlab list`) |

| Option | Meaning |
| --- | --- |
| `--compare torch,jacobi` | Reference methods shown next to yours; they do not affect the exit code |
| `--device cuda` | `cpu` (default), `cuda` or `cuda:N`; ROCm builds use the same spelling |
| `--quick` | Six smaller matrices for a fast first run |
| `--case NAME` | Run only this case; repeatable |
| `--rank K` | Ask every case for rank K instead of its own |
| `--repeat N`, `--warmup N` | Timed calls per case (median and MAD reported) and untimed warm-up calls |
| `--threads N` | CPU threads (default 4, so CPU timings are comparable between machines) |
| `--csv FILE` | Save every row and every metric; `.json` also works |
| `--gpu-check` | Run on CPU and on the GPU. A case passes when both runs are `ok`, the outputs are on the GPU and the singular values agree |

Exit codes: 0 when every case of every checked method is `ok`; 1 when one fails; 2 when a
method cannot be loaded or built (missing file, missing function, compiler error).

## 3. Replacing one step of an existing SVD

`svdlab.blocks` contains the steps, one matrix at a time. They are the repository's own
implementations (no vendor factorization), run on CPU and GPU, real and complex:

| Block | Returns |
| --- | --- |
| `qr(a)` | Householder QR of a tall matrix: `Q (m, n)`, `R (n, n)` |
| `torch_qr(a)` | The same from the library, as a control |
| `bidiagonalize(a)` | `Q, d, e, P` with `a = Q B Pᴴ`, B upper bidiagonal (real d, e) |
| `bidiagonal_svd(d, e)` | `U, s, V` with `B = U diag(s) Vᵀ` (implicit-shift QR, bdsqr) |
| `jacobi_svd(a, rank)` | One-sided Jacobi SVD of any matrix: `U, s, Vh` |
| `tridiagonalize(h)` | `Q, d, e` with Hermitian `h = Q T Qᴴ` |
| `hermitian_eig(h)` | Jacobi eigendecomposition: `w` ascending, `V` |
| `cholesky(g)` | Lower `L` with `g = L Lᴴ` |
| `solve_lower(l, b)`, `solve_upper(u, b)` | Triangular solves |
| `random_sketch(a, columns, seed)` | Gaussian test matrix for randomized methods |
| `complete_basis(q, size)` | Extends orthonormal columns |
| `svd(a, algorithm, rank)` | Any of the 16 repository SVDs (`blocks.ALGORITHMS`) |
| `torch_svd(a, rank)` | Library SVD |

`svdlab.recipes` assembles them into complete SVDs. Every step is a keyword argument:

```python
recipes.golub_kahan(a, rank, bidiagonalize=..., bidiagonal_svd=...)
recipes.qr_then_svd(a, rank, qr=..., small_svd=...)
recipes.randomized(a, rank, oversample=8, power_iterations=2, seed=0, sketch=..., qr=..., small_svd=...)
recipes.jacobi(a, rank)
```

Your replacement keeps the contract of the block it replaces. Wide matrices are
factorized through the adjoint, so tall-only steps never see a wide input. Example:
[examples/replace_a_step.py](../examples/replace_a_step.py) puts CholeskyQR2 inside
randomized SVD; the check then shows where that trade-off breaks.

## 4. C++ and CUDA

A `.cpp`/`.cc`/`.cu` file exports the same function through pybind11:

```cpp
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(torch::Tensor a, int64_t rank) { ... }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("svd", &svd); }
```

`python -m svdlab check my_svd.cu --device cuda` compiles it with
`torch.utils.cpp_extension.load`, `.cu` with nvcc, and caches the build in
`~/.cache/torch_extensions` (or `TORCH_EXTENSIONS_DIR`). It rebuilds when the file changes.
From Python: `svdlab.native.load('my_svd.cu')` returns the callable.

- [examples/native/my_svd.cpp](../examples/native/my_svd.cpp) uses ATen tensor operations,
  so the same code runs on CPU and GPU tensors.
- [examples/native/my_svd.cu](../examples/native/my_svd.cu) has a hand-written CUDA kernel
  (one thread block per column pair) for real float32/float64 input. The complex case
  reports an error, which is expected.

Requirements: `pip install ninja` (or `pip install -e '.[native]'`), the Python headers
(Debian/Ubuntu `sudo apt install python3-dev`), a C++ compiler and, for `.cu`, the CUDA
toolkit's nvcc (found through `CUDA_HOME`, default `/usr/local/cuda`). When something is
missing, the command lists it and exits with code 2.

## 5. From Python, and your own matrices

```python
from svdlab import check, compare, table, summary, save, MatrixCase, QUICK_SUITE
rows = check(svd, device='cuda')                       # one dict per matrix
rows = compare({'mine': svd, 'torch': 'torch'}, QUICK_SUITE)
table(rows)                                            # pandas DataFrame (pip install -e '.[laboratory]')
save(rows, 'laboratory_runs/mine.csv')
```

A case is a dict of `MatrixCase.create` arguments or a `MatrixCase`:

```python
cases = [
    dict(name='tall', m=500, n=200, family='gaussian', dtype='float32'),
    dict(name='low_rank', m=800, n=400, family='low_rank', rank=20, tail=1e-6, output_rank=20),
    MatrixCase.from_spectrum([5, 5, 1, 1e-3, 0], m=10, n=8),     # chosen singular values
    MatrixCase.load('matrix.npy'),                               # your own matrix
]
```

Families: `gaussian`, `low_rank` (`rank`, `tail`), `clustered` (`cluster_size`,
`cluster_spread`, `condition`), `geometric` (`condition`), `sparse` (`density`), `zero`.
`output_rank` is the rank requested from the method. `python -m svdlab list` prints the
standard sets. See [examples/from_python.py](../examples/from_python.py).

## 6. How the verdict is decided

Every metric is computed after the timed calls, in FP64 against a LAPACK `gesvd` reference
of the exact rounded input. The tolerance is `tol = 100 · eps(dtype) · max(m, n)`.

- Full SVD (`rank == min(m, n)`): `ok` when both orthogonality errors and the
  reconstruction error are `<= tol` and the singular values are non-negative and sorted.
- Truncated SVD: orthogonality `<= tol` and residual `<= 1.05 ×` the optimal rank-k
  residual (`tail×`).
- Values only: largest singular value error `<= tol · σ₁`.

Timing is the median of `--repeat` calls after `--warmup` calls, with the input already on
the device; on the GPU the device is synchronized around each call and CUDA-event time and
the Torch allocator peak are reported too. Matrix generation, the reference SVD and the
metrics are not timed.

## 7. Tests

```bash
python -m unittest discover -s tests                      # CPU
SVD_TEST_DEVICE=cuda python -m unittest discover -s tests # GPU
python -m unittest tests.test_svdlab -v                   # svdlab only
```

The native tests are skipped, with the reason printed, when build tools are missing.

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Python headers ... Python.h` missing | `sudo apt install python3-dev` (needed for C++/CUDA builds; on recent torch also for Triton, which some GPU matrix products use) |
| `ninja` missing | `pip install ninja` in the same environment |
| `nvcc` missing | Install the CUDA toolkit or `export CUDA_HOME=/usr/local/cuda` |
| `requested but torch.cuda.is_available() is False` | CPU-only torch build; install the CUDA build of PyTorch |
| `output_device` is `cpu` on a GPU run | The method moved data to the host; keep the computation on `a.device` |
| A slow Python loop on the full set | Start with `--quick` or `--case NAME`, then the full set |

## 9. Going deeper

- Trace the inside of an algorithm (checkpoints, intermediate arrays, iteration
  diagnostics, parameter sweeps): [algorithm laboratory](ALGORITHM_LABORATORY.md).
- Interactive comparisons in the browser: [playground](PLAYGROUND.md).
- Large benchmarks with memory and end-to-end timing: [benchmark guide](BROAD_BENCHMARK.md).
