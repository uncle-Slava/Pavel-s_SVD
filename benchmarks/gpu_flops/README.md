# GPU timing and FLOP benchmark of every SVD

Times every SVD in this repository on a CUDA GPU at n = 32, 64, 128, 256. That covers the 17 standalone solvers behind `svdlab.blocks.svd`, the `svdlab.recipes` golub_kahan, qr_then_svd, randomized and lanczos, and `torch.linalg.svd` as the reference. Each run records time, accuracy, iteration counts and executed floating-point operations.

## What is measured

- **Matrices:** one square Gaussian matrix per size, drawn in float64 with seed `1000 + n` and cast to float32 (float64 for a few methods). Every method gets `rank = n`, so the truncated methods (randomized, subspace, block_krylov, lanczos) also return a full SVD.
- **Time:** 3 warm-up runs, then 20 timed runs measured with CUDA events after `torch.cuda.synchronize()`. Median and minimum are reported.
  - A method whose first run takes over 2 s gets 1 warm-up and 3 timed runs.
  - A method slower than 5 s at n = 128 is not run at n = 256.
- **Accuracy:** computed in float64 from the returned factors:
  - reconstruction error ‖A − U diag(s) Vᵀ‖_F / ‖A‖_F;
  - orthogonality errors ‖UᵀU − I‖_F and ‖V Vᵀ − I‖_F;
  - maximum relative singular-value error against `torch.linalg.svdvals` in float64.
- **Iterations:** the solver's own diagnostics (`return_info=True`), plus counters wrapped around `primitives.jacobi_svd`, `jacobi_eigh` and `bdsqr`. These give Jacobi sweeps, bdsqr iterations, polar iterations and Lanczos steps.
- **FLOPs** (`flops_svd.py`): executed operations, with one multiply or add counted as one FLOP.
  - Matrix-multiply FLOPs are measured with `torch.utils.flop_counter.FlopCounterMode`.
  - The repository's Householder QR, bidiagonalization, tridiagonalization, Cholesky, triangular solves, one-sided and Hermitian Jacobi, and bidiagonal QR use element-wise kernels, not matmul. Their FLOPs are counted exactly from each call's shape and its actual sweep or iteration count, masked work included. Nothing is counted twice.
  - Not counted: the float64 secular-equation solves of the divide-and-conquer methods, norms, sorting and sketch generation.
  - `torch.linalg.svd` is a vendor kernel, so its FLOPs are the textbook estimate 21n³ (Golub–Reinsch with U and V), not a count.

## How to run

From the repository root:

```bash
.venv/bin/python benchmarks/gpu_flops/bench_svd.py --sizes 32 64 128 256 --device cuda --out benchmarks/gpu_flops/out
.venv/bin/python benchmarks/gpu_flops/flops_svd.py --sizes 32 64 128 256 --device cuda --out benchmarks/gpu_flops/out
cp benchmarks/gpu_flops/results/2026-10-30-gb10/algorithms.json benchmarks/gpu_flops/out/   # descriptions read by prep.py
python3 benchmarks/gpu_flops/prep.py benchmarks/gpu_flops/out benchmarks/gpu_flops/out/report_data
.venv/bin/python benchmarks/gpu_flops/plot_results.py benchmarks/gpu_flops/out
python3 benchmarks/gpu_flops/make_pdf.py benchmarks/gpu_flops/out   # needs Chromium (or set CHROME)
```

`prep.py` also reads `scorecard_cuda_table.json` (from `python -m svdlab check examples/my_svd.py --device cuda --compare torch,jacobi --csv .../scorecard_cuda_table.json`) and, optionally, `scorecard_gpu_check.txt` (stdout of the same command with `--gpu-check`).

For a quick check, use `--sizes 32 --repeat 3 --warmup 1`. Memory use is small: the largest case is a 256 × 256 matrix. The full run takes about an hour on a GB10 because the bidiagonal QR and block Jacobi solvers are slow; their n = 256 cases are skipped automatically.

## Results: NVIDIA GB10, 30 October 2026

Setup: PyTorch 2.14.1+cu130, CUDA 13.0. Files are in `results/2026-10-30-gb10/`:

- `results.json`: one record per method, size and dtype;
- `flops.json`: FLOPs broken down by primitive;
- `algorithms.json`: algorithm descriptions;
- `env.json`;
- `scorecard_*`: output of the standard `svdlab check`;
- `report_data/`: the tables behind the report;
- `report.pdf`: static report (findings, algorithm diagrams, charts, all tables), readable on GitHub;
- `report.html`: source of the interactive report (runs only in the claude.ai Dashboard runtime);
- charts: `time_vs_n`, `flops_vs_n`, `latency_vs_error` (`.svg` and `.png`).

Median time in ms, float32:

| method | n=32 | 64 | 128 | 256 | GFLOP at 256 | GFLOP/s at 256 | recon. error at 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| torch.linalg.svd | 0.23 | 1.06 | 2.67 | 7.16 | 0.35 (est.) | 49 | 6.4e-5 |
| polar_polynomial | 66 | 151 | 342 | 691 | 3.55 | 5.1 | 1.0e-4 |
| polar_newton_schulz | 66 | 151 | 343 | 692 | 3.52 | 5.1 | 1.0e-4 |
| polar_cans | 68 | 153 | 344 | 694 | 4.46 | 6.4 | 9.9e-5 |
| gram_dc | 83 | 174 | 357 | 732 | 0.29 | 0.4 | 5.5e-4 |
| gram_jacobi | 65 | 150 | 416 | 769 | 1.74 | 2.3 | 5.5e-4 |
| bidiag_dc | 91 | 190 | 391 | 857 | 0.57 | 0.7 | 6.2e-7 |
| polar_qdwh | 90 | 199 | 440 | 883 | 2.30 | 2.6 | 1.0e-4 |
| jacobi | 72 | 163 | 403 | 890 | 1.80 | 2.0 | 5.5e-5 |
| lanczos | 77 | 156 | 426 | 927 | 2.14 | 2.3 | 4.0e-5 |
| qr / tsqr / recipes.qr_then_svd | 78 | 175 | 426 | 935 | 1.89 | 2.0 | 5.5e-5 |
| block_krylov | 90 | 200 | 442 | 1027 | 2.14 | 2.1 | 5.5e-5 |
| polar_halley | 96 | 220 | 496 | 1057 | 2.51 | 2.4 | 1.0e-4 |
| randomized / subspace / recipes.randomized | 103 | 225 | 528 | 1127 | 2.32 | 2.1 | 5.5e-5 |
| bidiag_qr ≈ recipes.golub_kahan | 597 | 2272 | 8712 | skipped | | | |
| recipes.lanczos | 594 | 2261 | 8600 | skipped | | | |
| block_jacobi | 1128 | 5875 | 31828 | skipped | | | |

Findings:

- **Speed:** every repository solver is limited by kernel launches and host synchronization, not by arithmetic. The best reaches about 6 GFLOP/s, against 49 GFLOP/s for cuSOLVER. Time grows roughly linearly in n (rounds × sweeps of small kernels) instead of as n³. At n = 256 the fastest repository SVDs are about 97× slower than `torch.linalg.svd`.
- **Slowest:** bidiag_qr and block_jacobi are an order of magnitude slower than the rest.
  - bidiag_qr: bdsqr is a Python loop over the bidiagonal with Givens rotations each iteration (163 iterations at n = 128).
  - block_jacobi: it calls the inner Jacobi solve 4960 times at n = 128.
- **Accuracy (float32):** bidiag_dc reconstructs best (6e-7, float64 merges), then bidiag_qr (2e-6). The Jacobi family is at about 5e-5 and the polar family at about 1e-4.
- **Gram methods:** gram_jacobi and gram_dc square the condition number, so at n = 256 they lose the smallest singular values (relative σ error 1.0).
- **Failures:** none. polar_cans converges (17–28 polar iterations).

Caveats:

- The GPU was shared with another, idle process.
- The σ error of `torch.linalg.svd` in float64 is 0 by construction, since it is also the reference.
- Timings include Python overhead, which is the point of comparing these implementations as written.
