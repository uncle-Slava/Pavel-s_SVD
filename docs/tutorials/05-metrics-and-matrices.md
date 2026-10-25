# Tutorial 5: Read the metrics and test your own matrices

**What this tutorial is about.** The report shows a verdict and four short numbers per
matrix. Behind them are about twenty measurements. This tutorial explains each one, the
exact rule that decides `ok` or `FAIL`, and what that rule deliberately does not check.
Then it shows how to build test matrices that target the weak points of your method, how
to save and analyze results, and what the GPU check adds.

**You will learn:**

- where the numbers come from and what is (and is not) timed;
- the formula behind every metric, and when a case counts as `ok`;
- why the verdict alone may not be enough for your application;
- how to create test matrices by family, by chosen singular values, or from a file;
- how to save results and compare methods with pandas;
- what the GPU check catches that a plain GPU run does not.

**Before you start:** [tutorial 1](01-install-and-first-check.md). This tutorial is also a
reference: come back to it whenever a number in a report is unclear.

**Notebook version:** [notebooks/05_metrics_and_matrices.ipynb](../../notebooks/05_metrics_and_matrices.ipynb)
(recomputes the metrics by hand, so you can see there is nothing hidden).

## 5.1 Where the numbers come from

For every test matrix `svdlab`:

1. generates the matrix from a fixed seed (or loads it), so every run and every machine
   sees the same bytes;
2. computes an independent reference SVD of that exact input with LAPACK `gesvd` in
   float64 (complex128 for complex input);
3. calls your method once, converts its result to float64 and computes all metrics;
4. calls it again `--warmup` times untimed and `--repeat` times timed.

Steps 1, 2 and 3 are not timed. The time is only your function, with the input already on
the device.

## 5.2 The accuracy metrics

Let `k` be the requested rank, `A` the input, and `U`, `s`, `Vh` your result cut to `k`
triplets. All norms are Frobenius norms.

| Short name | Column in `table(rows)` / CSV | Formula | What it tells you |
| --- | --- | --- | --- |
| `recon` | `reconstruction_relative_fro` | ‖A − U diag(s) Vh‖ / ‖A‖ | Do the factors rebuild A? For a truncated SVD it cannot be smaller than the part of A that is left out. |
| `orth` | `orthogonality_u_fro_over_sqrt_cols`, `orthogonality_v_fro_over_sqrt_rows` | ‖UᴴU − I‖ / √k, ‖Vh Vhᴴ − I‖ / √k | Are the singular vectors orthonormal? The printed `orth` is the larger of the two. |
| `tail×` | `residual_over_optimal_tail` | `recon` / ‖σ_{k+1..}‖ / ‖A‖ | Truncated cases only. The best possible rank-k residual (Eckart–Young) is ‖(σ_{k+1}, σ_{k+2}, ...)‖; 1.00 means optimal. |
| `σ-rel` | `resolved_spectrum_max_relative_error` | max over i of \|s_i − σ_i\| / σ_i | Relative accuracy of each singular value. Values below the float64 noise floor (eps · max(m, n) · σ₁) are skipped, because the reference itself is not accurate there. |
| | `spectrum_error_over_sigma1` | max \|s_i − σ_i\| / σ₁ | Absolute error of the values, relative to the largest one. Used for the verdict when only `s` is returned. |
| | `left_triplet_relative_fro`, `right_triplet_relative_fro` | ‖A V − U diag(s)‖ / ‖A‖, ‖Aᴴ U − V diag(s)‖ / ‖A‖ | Does each triplet satisfy the defining equations A v = σ u and Aᴴ u = σ v? |
| | `right_subspace_leakage_fro_over_sqrt_k` | ‖V − V_ref V_refᴴ V‖ / √k | How much of your top-k right subspace lies outside the true one. Only when there is a clear gap after σ_k (otherwise the subspace is not well defined). |
| | `singular_values_nonnegative_sorted` | | `s >= 0` and descending. |

Signs and phases of singular vectors are never compared: (u, v) and (−u, −v) are equally
correct, and inside a cluster of equal singular values any orthonormal basis is correct.
That is why all metrics are based on residuals, orthogonality and subspaces.

## 5.3 When a case is `ok`

The tolerance depends on the input precision and size:

```text
tol = 100 · eps(dtype) · max(m, n)        eps = 2.2e-16 (float64, complex128), 1.2e-7 (float32, complex64)
```

For a 48×32 float64 matrix that is 100 · 2.2e-16 · 48 ≈ 1.1e-12. For a 400×300 float32
matrix it is about 4.8e-3.

| Case type | `ok` when |
| --- | --- |
| Full SVD (`rank == min(m, n)`) | values sorted and non-negative, both `orth` ≤ tol, `recon` ≤ tol |
| Truncated SVD (`rank < min(m, n)`) | values sorted and non-negative, both `orth` ≤ tol, `tail×` ≤ 1.05 |
| Only `s` returned | values sorted and non-negative, `spectrum_error_over_sigma1` ≤ tol |

What the verdict deliberately does **not** require:

- **Relative accuracy of tiny singular values.** On `ill_conditioned_1e8` a correct
  backward-stable method has `σ-rel` around 1e-10 (tutorial 1). Requiring 1e-15 would fail
  every standard algorithm.
- **Accurate singular values in a truncated case.** The verdict asks for a near-optimal
  approximation. Example: randomized SVD with rank 5 on a 60×40 Gaussian matrix (whose
  spectrum does not decay at all) gives `tail× 1.00`, so it is `ok`, but `σ-rel` is 1.4e-2.
  On a matrix with a fast-decaying spectrum the same method gives `σ-rel` 1.4e-15. If your
  application needs accurate values, look at `σ-rel` in addition to the verdict.

The verdict is a minimum bar for correctness, not a ranking. To compare two correct
methods, compare their metrics and times.

## 5.4 Timing columns

| Column | Meaning |
| --- | --- |
| `time_median_ms` | median wall time of `--repeat` calls (default 5) after `--warmup` calls (default 1); on the GPU the device is synchronized before and after each call |
| `time_mad_ms` | median absolute deviation of those times: a large value means noisy timings, so increase `--repeat` |
| `device_time_median_ms` | GPU only: median time measured with CUDA events |
| `gpu_peak_increment_bytes` | GPU only: peak memory allocated through the PyTorch allocator during a call, above what was allocated before |
| `output_device` | where the result came back: `cpu`, `cuda:0`, `numpy` or a mix such as `cpu,cuda:0` |

CPU runs use 4 threads by default (`--threads N`), so CPU timings are comparable between
machines with different core counts. TF32 matrix products are switched off on the GPU, so
float32 means real float32.

## 5.5 Your own matrices

The standard sets are printed by `python -m svdlab list`. A test case is a `dict` of
generator arguments, or a `MatrixCase`:

```python
from svdlab import check, MatrixCase

cases = [
    dict(name='tall_fp32', m=500, n=200, family='gaussian', dtype='float32'),
    dict(name='cond_1e12', m=200, n=200, family='geometric', condition=1e12),
    dict(name='pairs', m=120, n=80, family='clustered', cluster_size=2, cluster_spread=1e-10),
    dict(name='rank20', m=800, n=400, family='low_rank', rank=20, tail=1e-6, output_rank=20),
    MatrixCase.from_spectrum([5, 5, 1, 1e-3, 1e-9, 0], m=10, n=8),   # exactly these singular values
    MatrixCase.load('my_matrix.npy'),                                 # your own data
]
cases[-1].spec['name'] = 'my_matrix'          # otherwise a loaded matrix is named by its hash
rows = check(my_svd, cases)
```

| `family` | Singular values | Parameters |
| --- | --- | --- |
| `gaussian` | random (Marchenko–Pastur): well conditioned for clearly tall or wide shapes, can be badly conditioned when square | |
| `geometric` | from 1 down to 1/`condition`, geometrically | `condition` (default 1e6) |
| `clustered` | groups of `cluster_size` nearly equal values, group centers geometric | `cluster_size`, `cluster_spread` (relative width of a group), `condition` |
| `low_rank` | `rank` values from 1 to 0.1, then all equal to `tail` | `rank`, `tail` (default 1e-5) |
| `sparse` | Gaussian entries, a fraction `density` non-zero, stored dense | `density` (default 0.05) |
| `zero` | all zero | |

Common arguments: `m`, `n`, `dtype` (`float32`, `float64`, `complex64`, `complex128`;
default `float64`), `seed` (default 7), `name` (shown in the report) and `output_rank`
(the `rank` passed to your method; default `min(m, n)`, a full SVD). A `.npy` file must
hold one finite 2-D float32/float64/complex64/complex128 array.

Matrices that often break SVD methods, and the case that shows it:

| Weakness | Case |
| --- | --- |
| Forming AᴴA or YᴴY (Gram, CholeskyQR) | `geometric` with `condition` 1e8 or more |
| Deflation and convergence criteria | `clustered` with a tiny `cluster_spread`, or `from_spectrum` with repeated values |
| Exact zeros, rank deficiency | `from_spectrum([..., 0, 0])`, `zero` |
| Wide inputs | `n > m` |
| Complex arithmetic (missing `.conj()`) | `dtype='complex128'` |
| float32 accumulation | `dtype='float32'` and large `m` |

## 5.6 Save and analyze results

```python
from svdlab import compare, save, summary, table

rows = compare({'mine': my_svd, 'torch': 'torch'}, cases, repeat=5)
save(rows, 'laboratory_runs/mine.csv')       # or .json; laboratory_runs/ is ignored by git
df = table(rows)                             # pandas DataFrame with every column above
df.pivot(index='case', columns='method', values='time_median_ms')
summary(rows)                                # per method: ok count, failed cases, worst errors, total time
```

From the terminal: `python -m svdlab check my_svd.py --compare torch --csv results.csv`.

## 5.7 The GPU check

`--gpu-check` (or `svdlab.gpu_check(fn, cases)`) runs the same method on the CPU and on
the GPU and reports one row per case:

| Field | Meaning |
| --- | --- |
| `cpu_ok`, `gpu_ok` | the verdict on each device |
| `output_device`, `on_gpu` | where the GPU run returned its result |
| `max_singular_value_difference_over_sigma1` | max \|s_cpu − s_gpu\| / σ₁ |
| `speedup` | CPU median time / GPU median time |
| `passed` | `on_gpu` and `cpu_ok` and `gpu_ok` and the difference ≤ tol |

It catches the two GPU problems that a plain GPU run does not show: code that quietly
moves the data to the CPU (correct numbers, wrong timing) and a GPU path that is
numerically different from the CPU path. A method that draws random numbers from a
device generator (for example a randomized SVD seeded on the device) gets different
sketches on CPU and GPU; its singular values then agree only as far as the sketch does
not matter.

```bash
python -m svdlab check my_svd.py --gpu-check --quick
```

## 5.8 Using it in CI

The exit code is 0 when every case of the checked methods is `ok`, 1 when one fails and
2 when a method cannot be loaded or compiled. Methods given with `--compare` never change
the exit code. A minimal CI step:

```bash
python -m pip install torch && python -m pip install -e .
python -m svdlab check my_svd.py --quick --csv svdlab-results.csv
```

## Next

- [Developer guide](../DEVELOPER_GUIDE.md): the compact reference.
- [Algorithm laboratory](../ALGORITHM_LABORATORY.md): look inside an algorithm while it
  runs (intermediate matrices, per-iteration errors, parameter sweeps).
