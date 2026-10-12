# Standalone SVD laboratory

This package implements every family in the preceding research table, plus incremental
updates and tensor decompositions. It is an executable mathematical baseline for changing
algorithms and checking their accuracy. It uses PyTorch tensor storage, matrix multiplication,
reductions, indexing and elementwise operations. **The standalone package never calls a
vendor SVD, QR, eigensolver, Cholesky, triangular solve or inverse.** Matrix multiplication
can still use the framework's BLAS backend. It is not an implementation of PyTorch itself.

The production code is in `svdbench/standalone/` and does not import the older Hetzner
project. NumPy/SciPy are used outside it for input generation and independent validation.
There are no compiled extensions to build. The existing `/root/svdbench` project is unchanged.
The new deployment is `/root/svd-standalone-20261011`.

## Implementations

| Selector/API | Implemented computation | Relevant qualification |
|---|---|---|
| `bidiag_qr` | Householder bidiagonalization, implicit-shift Golub–Kahan QR, deflation, vector back-transformation | Per-batch active windows; zero-shift handling; raises on iteration exhaustion |
| `bidiag_dc` | Recursive bidiagonal row split, diagonal-plus-rank-one secular merges, vector back-transformation | Real merge stage uses FP64 on the same device; guarded standalone Jacobi merge refinement is counted |
| `jacobi` | One-sided complex/real plane rotations in a round-robin schedule | Independent column pairs share each round; all batch members advance together |
| `block_jacobi` | Block-pair orthogonalization with a local standalone Jacobi SVD | Python block orchestration; not a fused GPU block kernel |
| `qr` | Householder QR followed by standalone Jacobi SVD of R | Reduced storage; supports tall and wide inputs |
| `tsqr` | Row-tile QR, binary tree of stacked-R QR factorizations, back-transformation | Explicit tree; currently one device, with resident row tiles |
| `gram_jacobi` | Smaller Gram matrix, Hermitian Jacobi eigensolve, singular-vector recovery | Deliberately exposes condition-number squaring |
| `gram_dc` | Smaller Gram matrix, Householder tridiagonalization, Cuppen D&C, secular merges | Distinct from QR/QL; same Gram accuracy limitation |
| `polar_qdwh` | Dynamically weighted Halley iteration, stacked QR for large weights, Cholesky/triangular solves for small weights, Hermitian Jacobi finish | No preliminary vendor spectral solve; lower-bound estimate is configurable |
| `polar_newton_schulz` | Scaled Newton–Schulz polar iteration, Hermitian Jacobi finish | Cubic matrix polynomial per iteration; slow for tiny nonzero values |
| `polar_polynomial` | Fifth-degree polar iteration X(15I−10G+3G²)/8, G=X*X, then Hermitian Jacobi | Convergent inverse-square-root polynomial on the normalized range; not the empirical optimizer coefficients from the old project |
| `polar_halley` | Fixed-weight Halley polar iteration using standalone Cholesky/solves, Hermitian Jacobi finish | Useful control for the dynamically weighted version |
| `randomized` | Gaussian range finder, stabilized power steps, projected standalone Jacobi solve | Oversampling, power count and sketch seed configurable |
| `subspace` | The same stabilized block iteration with optional previous right basis | Shared numerical kernel is intentional; `initial_basis` enables a real warm start |
| `block_krylov` | Concatenated block Krylov range, double projection, Householder reorthogonalization, projected solve | More storage and passes than a single-block range finder |
| `lanczos` / `lanczos_svd` | Golub–Kahan bidiagonalization, full double reorthogonalization, breakdown restart, projected standalone solve | Callback API supports sparse/matrix-free operators; partial results need residual checks |
| `rank_one_update` | Augmented left/right spaces and small-core SVD for A+xy* | Updates the matrix represented by the supplied factors; old truncation error cannot be recovered |
| `append_columns` | Residual column space plus small-core SVD | Handles one or several appended columns |
| `tucker` | HOSVD or sequentially truncated HOSVD | Reports unfolding shapes, bytes and storage sharing |
| `tensor_train` | Successive unfoldings, truncated SVDs, propagation of S V* | Returns reconstructable TT cores and achieved bond ranks |

These are independently editable realizations of the public algorithm families, **not copies
of NVIDIA's private kernels or exact reproductions of LAPACK/rocSOLVER internals**. In
particular, D&C uses pole-relative bisection instead of the optimized rational secular solver
and performs a disclosed Jacobi refinement when a merge loses orthogonality or residual
accuracy. Both the secular merge count and refinement count are returned. Difficult cases
can therefore be substantially slower than a pure secular path.

The mathematical D&C structure follows the rank-one merge described in
[LAPACK DLASD1](https://www.netlib.org/lapack/explore-html/db/da1/group__lasd1_ga14c55d168287f716d70c17a2dc000268.html).
The QDWH iteration follows [Nakatsukasa, Bai and Gygi](https://epubs.siam.org/doi/10.1137/090774999).
Randomized range finding and stabilized power iteration follow
[Halko, Martinsson and Tropp](https://arxiv.org/abs/0909.4061).
The implementations and measured behavior, rather than those sources, define this package's
particular parameter choices and accuracy envelope.

## Using the algorithms

```python
import torch
from svdbench.standalone import svd

A = torch.randn(256, 32, device="cuda:0", dtype=torch.float32)
U, s, Vh, info = svd(A, "bidiag_dc", return_info=True)
U, s, Vh = svd(A, "randomized", rank=8, oversample=8, iterations=2, seed=31)
U, s, Vh = svd(A, "tsqr", tile_rows=64)

# Arbitrary leading batch dimensions, real or complex.
batch = torch.randn(4, 8, 16, 16, device="cuda:0", dtype=torch.complex64)
U, s, Vh = svd(batch, "jacobi")

# Square bases for a rectangular matrix; values-only is also accepted.
U, s, Vh = svd(A, "qr", full_matrices=True)
_, s, _ = svd(A, "bidiag_qr", values_only=True)
```

Values-only currently forms singular vectors internally and discards them. Its timing is
not an optimized values-only algorithm. Numerical null columns are completed into orthogonal
bases. Tiny values below a method's declared numerical floor are zeroed; inspect small-value
errors as well as reconstruction. Inputs are scaled before dense factorization to reduce
underflow/overflow and are never overwritten. Autograd is not supported.

```python
from svdbench.standalone.iterative import rank_one_update, append_columns

Un, sn, Vhn = rank_one_update(U, s, Vh, x, y, rank=8)
Un, sn, Vhn = append_columns(U, s, Vh, new_columns, rank=12)
```

Pass **reduced** consistent factors into the update APIs, not the full rectangular null bases.
Warm starts use `svd(A, "subspace", rank=k, oversample=p, initial_basis=V_previous)`;
the basis must have k+p columns, capped at min(m,n), and match the batch shape. Seeded
sketches are reproducible within a runtime/device; bitwise equality across different backend
random-number implementations is not promised.

For matrix-free work, provide `shape`, `dtype`, `device`, `matmat(X)` and `rmatmat(X)` to
`lanczos_svd(operator, rank=k, steps=j)`. `rmatmat` must apply the conjugate transpose. The
public matrix-free API expects sensibly scaled operator products; the dense `svd` wrapper
performs input scaling. This version uses full reorthogonalization and stores both bases.
It is not an implicitly restarted production sparse solver.

```python
from svdbench.standalone.tensors import (
    tucker, tucker_reconstruct, tensor_train, tensor_train_reconstruct,
)
core, factors, info = tucker(X, [4, 3, 2], sequential=True)
X_approx = tucker_reconstruct(core, factors)
cores, info = tensor_train(X, [3, 3])
X_approx = tensor_train_reconstruct(cores)
```

## Reproducing the Hetzner run

The existing Python environment supplies the runtime. Nothing needs to be installed globally:

```bash
cd /root/svd-standalone-20261011
export PYTHONPATH=/root/svd-research-20261011/vendor
PY=/root/svdbench/.venv/bin/python

SVD_TEST_DEVICE=cuda "$PY" -m unittest discover -s tests -v
"$PY" tools/run_monitored.py --telemetry results/my-telemetry.csv \
  --config configs/standalone.json --device cuda:0 \
  --output results/my-matrix-run.jsonl --corpus corpus/standalone \
  --warmup 1 --repeat 3 --end-to-end
"$PY" -m svdbench --config configs/standalone_scale.json --device cuda:0 \
  --output results/my-scale-run.jsonl --corpus corpus/standalone \
  --warmup 1 --repeat 3
"$PY" tools/benchmark_standalone_apps.py --device cuda:0 \
  --output results/my-app-run.jsonl --repeat 3
```

Output files are created exclusively to preserve prior runs. Run the same matrix config
with `--device cpu` for a host baseline. A custom experimental solver can be added to
`ALGORITHMS`/`_raw`, or supplied through the existing `module:factory` benchmark adapter.
The factory returns a callable yielding `(U,s,Vh)` and can expose `metadata` and `diagnostics`.

## Validation and measurement contract

The tests exercise all registered matrix solvers across FP32/FP64/complex64/complex128,
tall/wide/batched shapes, singleton dimensions, zero/rank-deficient/repeated spectra,
extreme scaling, full bases, values-only output, ill-conditioning, truncated approximation,
warm starts, sparse callbacks, updates, and tensor reconstruction. Independent SciPy FP64
or complex128 references assess actual rounded inputs. Numerical tests include iteration
exhaustion and invalid-input failures. A runtime test disables framework factorizations, and
an AST audit disallows `linalg` access or NumPy/SciPy/ctypes imports in the standalone code.

A passing execution or `converged=True` means the iteration finished, **not** that every
accuracy budget is satisfied. Gram methods may lose vector orthogonality on ill-conditioned
inputs even after dropping unresolved values. Partial SVDs should be assessed against the
best rank-k tail, not against a full reconstruction target. Repeated singular values make
individual vectors nonunique; use invariant-subspace or reconstruction comparisons.

Timings use warmup, synchronization, wall clocks and CUDA events; no graph replay. Python
control, scalar convergence checks, allocation, normalization, and sketch generation are
included in the standalone resident solve. End-to-end timing additionally includes input
copy, preparation and output copy. Corpus construction and CPU reference solves are outside
the timed region. Event spans can include gaps while the host dispatches work; they are not
the sum of kernel execution durations. Three repetitions provide a pilot, not a statistically
robust speed claim.

Memory reports distinguish input/output bytes from incremental PyTorch allocator peak.
D&C's FP64 promotion and temporary storage are included. Telemetry is coarse whole-device
sampling; it does not establish per-algorithm energy, occupancy, bandwidth or native library
workspace sizes. The application benchmark excludes the initial factorization so update
costs are identifiable; it includes a separate vendor full-recompute control.

## Porting and next optimization work

The mathematical operations and validation harness are shared. Port `primitives.py` first:
reflectors, reductions, rotations, triangular substitution and GEMM. Next fuse each small
Jacobi block/sweep, QR panel, secular root batch or polar step. Preserve failure indicators,
scaling and rank decisions. Replace host convergence checks with device counters where it
improves measured performance. The existing Python loops intentionally expose dependencies,
so their launch overhead is not a prediction of native CUDA/HIP/Ascend C speed.

ROCm can execute the same tensor expressions where supported, but has not been tested here.
The Ascend310B needs capability checks before choosing dtypes and kernels; FP64 merges and
complex arithmetic cannot be assumed available. This package does not claim an automatic
Python-to-CUDA/HIP/Ascend C translator or tested native ports.

## Provenance

`primitives.py` started from the user-owned Hetzner file
`/root/svdbench/gpubench/custom/linalg.py`, read on 11 October 2026. Changes include reduced
Householder accumulation to avoid an m-by-m temporary, dtype-safe denominators, a singleton
QR-solver path, zero-shift initialization, explicit iteration failure, stable hypot-based
rotations, numerical-null detection, basis completion, and corrected Jacobi stopping tests.
The new solver orchestration, D&C module, matrix-free/update module, tensor module and
standalone tests live in this project. No binary reverse engineering was used.
