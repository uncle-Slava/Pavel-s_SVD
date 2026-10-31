# Tutorial 2: Write your own SVD

**What this tutorial is about.** You write an SVD yourself as one Python function and check
it. The example is deliberately a method that looks fine at first: it is short, fast and
correct on ordinary matrices, but it fails on two of the test matrices. You find out why,
look at the failure in detail, and fix it. That loop (write, check, understand the
failure, fix) is how you will work on your own methods.

**You will learn:**

- the exact rules your function has to follow (the *contract*);
- how to test a function on one matrix before the full check;
- why a good reconstruction error alone does not prove an SVD is correct;
- how to look at a failing case: compare with the reference, stop in the debugger;
- a common way wide matrices are handled (work on the adjoint);
- how to move from a notebook to a file that the command line checks.

**Before you start:** [tutorial 1](01-install-and-first-check.md), so that the installation
works and you can read a report line.

**Notebook version:** [notebooks/02_write_your_own_svd.ipynb](../../notebooks/02_write_your_own_svd.ipynb)
(with a plot of the failure and a cell for your own code).

## 2.1 The contract

An SVD method is a Python function with this signature:

```python
def svd(a, rank):
    ...
    return U, s, Vh
```

| Argument / result | Shape and type |
| --- | --- |
| `a` | `torch.Tensor` of shape `(m, n)`, `float32`, `float64`, `complex64` or `complex128`, already on the device being tested. It can be tall (`m > n`) or wide (`m < n`). **Do not modify it in place.** |
| `rank` | `int`, the number of singular triplets the case needs. `rank == min(m, n)` for a full SVD, smaller for the low-rank cases. |
| `U` | `(m, k)` with orthonormal columns, `k >= rank` |
| `s` | `(k,)`, real, non-negative, in **descending** order |
| `Vh` | `(k, n)`, the conjugate transpose of V, so that `a ≈ U @ diag(s) @ Vh` |

Rules that save time:

- Returning more than `rank` triplets is fine; the extra ones are cut off. A full SVD is
  therefore a valid answer for every case, so you can ignore `rank` at first.
- You may return torch tensors or NumPy arrays.
- If you return only `s` (one 1-D tensor), only the singular values are checked.
- On the GPU, return GPU tensors. If the result comes back on the CPU, the report shows
  it in `output_device` and `--gpu-check` fails the case, because the time would not be
  the time of a GPU computation.
- Inside the function you can use anything: PyTorch, NumPy, SciPy, a C++ extension.
  Use plain PyTorch operations if you want the same code to run on CPU and GPU.

The file can contain other functions and imports. `svdlab` uses the function named `svd`
(or `my_svd`). To check another function in the same file, write `file.py:function_name`.

## 2.2 A first SVD: the Gram matrix

The singular values of A are the square roots of the eigenvalues of the Gram matrix
AᴴA, and its eigenvectors are the right singular vectors V. Then U = A V / s. That gives a
very short SVD. It is in [examples/gram_svd.py](../../examples/gram_svd.py):

```python
import torch

def svd(a, rank):
    """SVD through the eigendecomposition of the Gram matrix AᴴA."""
    g = a.mH @ a                                   # (n, n) Hermitian, eigenvalues = s**2
    w, v = torch.linalg.eigh(g)                    # ascending
    w, v = w.flip(0), v.flip(1)                    # descending
    s = w.clamp_min(0).sqrt()
    u = (a @ v) / s.clamp_min(torch.finfo(s.dtype).tiny)
    return u[:, :rank], s[:rank], v[:, :rank].mH
```

Before running the full check, a one-matrix smoke test catches shape mistakes:

```python
a = torch.randn(6, 4, dtype=torch.float64)
u, s, vh = svd(a, 4)
print(u.shape, s.shape, vh.shape)                  # (6, 4) (4,) (4, 4)
print(torch.linalg.norm(a - u * s @ vh))           # about 1e-15
```

(`u * s` multiplies column j of `u` by `s[j]`; it is the same as `u @ torch.diag(s)`.)

## 2.3 Check it

```bash
python -m svdlab check examples/gram_svd.py --quick --compare torch
```

```text
gram_svd             gaussian_tall          ok       0.703 ms  recon 1.47e-15  tail× -  σ-rel 3.33e-15  orth 3.53e-15
gram_svd             gaussian_wide          ok       0.756 ms  recon 1.42e-15  tail× -  σ-rel 2.70e-15  orth 2.08e-15
gram_svd             ill_conditioned_1e8    FAIL     0.671 ms  recon 7.55e-16  tail× -  σ-rel 5.88e-01  orth 2.26e-01
gram_svd             clustered              FAIL     0.709 ms  recon 1.55e-15  tail× -  σ-rel 1.44e-11  orth 1.05e-11
gram_svd             complex_gaussian       ok       0.643 ms  recon 1.12e-15  tail× -  σ-rel 3.78e-15  orth 2.83e-15
gram_svd             low_rank_k4            ok       1.199 ms  recon 6.64e-06  tail× 1.00e+00  σ-rel 6.44e-16  orth 7.96e-16
...
gram_svd: 4/6 ok, total median 4.680 ms, failed: ill_conditioned_1e8, clustered
```

It is about 100 times faster than the Jacobi starter of tutorial 1, and it fails two cases.
The exit code is 1.

## 2.4 Understand the failure

Look at `ill_conditioned_1e8`. `recon` is excellent (7.6e-16), yet `σ-rel` is 0.59 and
`orth` is 0.23: the product U diag(s) Vᴴ rebuilds A, but U is far from orthonormal and the
small singular values are wrong by 59 %. A reconstruction test alone would have
accepted this answer. That is why the verdict also checks orthogonality.

The reason is numerical: forming AᴴA squares the condition number. This matrix has
singular values from 1 down to 1e-8, so AᴴA has eigenvalues from 1 down to 1e-16, which is
below float64 rounding (eps ≈ 2.2e-16). `eigh` cannot resolve them, and dividing by
those tiny, inaccurate `s` destroys the orthogonality of `U`. On `clustered` the same
squaring makes nearly equal singular values hard to separate, which costs about five
digits of orthogonality.

You can see it directly by comparing with the FP64 reference that `svdlab` uses:

```python
from svdlab import MatrixCase
case = MatrixCase.create(m=40, n=30, family='geometric', condition=1e8, dtype='float64')
u_ref, s_ref, vh_ref = case.reference[0]          # LAPACK gesvd of the exact input, in FP64
u, s, vh = svd(case.tensor('cpu'), 30)
print(s_ref[-5:])      # [1.27e-07 6.72e-08 3.56e-08 1.89e-08 1.00e-08]
print(s[-5:])          # [1.27e-07 6.64e-08 2.88e-08 2.66e-08 1.59e-08]  <- 1 % to 60 % wrong
```

Debugging tips:

- `--case ill_conditioned_1e8` runs only that matrix; `--repeat 1 --warmup 0` skips the
  repeated timing calls.
- Put `breakpoint()` in your function and run `python -m svdlab check my_svd.py --case NAME`:
  you stop inside your code with the real test matrix in `a`.
- An exception inside your function does not stop the check. The case is reported as
  `ERROR` with the exception message, and the other cases still run.

## 2.5 Fix it

Keep the fast Gram step, but use its eigenvectors only as a starting basis. The columns
of A V are then already nearly orthogonal, and a one-sided Jacobi SVD of A V finishes in
very few sweeps with full accuracy. `svdlab.blocks.jacobi_svd` is a ready-made Jacobi SVD
(tutorial 3 lists all the blocks):

```python
from svdlab import blocks

def refined(a, rank):
    """Gram eigenvectors as a starting basis, then Jacobi on A V to repair the accuracy."""
    wide = a.shape[0] < a.shape[1]
    x = a.mH if wide else a                        # work on a tall matrix
    _, v = torch.linalg.eigh(x.mH @ x)
    u, s, wh = blocks.jacobi_svd(x @ v)            # x V = U diag(s) Wᴴ, so x = U diag(s) (V W)ᴴ
    vh = wh @ v.mH
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    return (vh.mH, s, u.mH) if wide else (u, s, vh)
```

```bash
python -m svdlab check examples/gram_svd.py:refined --quick --compare examples/my_svd.py
```

```text
refined: 6/6 ok, total median 301.503 ms
my_svd: 6/6 ok, total median 584.612 ms
```

All cases pass, and it is about twice as fast as the plain Jacobi starter.

Note the pattern for wide matrices: factorize the adjoint `aᴴ = U' s V'ᴴ`, then
`a = V' s U'ᴴ`, so you return `(V', s, U'ᴴ)`. Many algorithms are simpler to write for tall
matrices only.

## 2.6 From Python instead of the terminal

Everything above also works from Python, which is handy in a notebook:

```python
from svdlab import check, compare, summary, table, QUICK_SUITE

rows = check(svd, QUICK_SUITE)                    # prints one line per case, returns a list of dicts
rows = compare({'gram': svd, 'refined': refined, 'torch': 'torch'}, QUICK_SUITE, repeat=3)
table(rows)                                       # pandas DataFrame, every metric
summary(rows)                                     # per method: cases passed, worst errors, total time
```

`check` and `compare` accept a function, a file name (`'examples/gram_svd.py:refined'`)
or a builtin name (`'torch'`). Options: `device='cuda'`, `rank=K`, `repeat=N`,
`warmup=N`, `verbose=False`.

## 2.7 Your own file

```bash
cp examples/my_svd.py my_svd.py          # or examples/gram_svd.py
# edit the body of svd(a, rank)
python -m svdlab check my_svd.py --quick
python -m svdlab check my_svd.py --compare torch          # the full set of 8 matrices
```

Work through `--quick` until everything is `ok`, then run the full set. Python loops over
single matrix elements are slow; loops over columns or over groups of columns with
tensor operations inside, like the Jacobi starter, are fine.

## Next

- [Tutorial 3](03-blocks-and-recipes.md): build an SVD from ready-made steps.
- [Tutorial 5](05-metrics-and-matrices.md): every metric, and your own test matrices.
