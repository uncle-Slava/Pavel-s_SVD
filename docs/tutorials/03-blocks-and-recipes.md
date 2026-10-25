# Tutorial 3: Build an SVD from blocks and recipes

**What this tutorial is about.** Most SVD algorithms are a chain of a few standard steps:
reduce the matrix to a simpler form (a QR factorization, a bidiagonal matrix), solve the
small or simple problem, and map the result back. Often you only want to improve one of
those steps, for example a faster QR on the GPU, and keep the rest. `svdlab` gives you the
steps (*blocks*) and complete algorithms built from them (*recipes*), in which each step is
a parameter you can replace.

**You will learn:**

- which blocks exist and what each one takes and returns;
- how the five recipes are put together, and what each is good for;
- how to plug your own step into a recipe, and how the check shows its trade-offs;
- how to find the slowest step of an algorithm by swapping in library versions;
- how to tune a recipe's parameters, and how to write a new chain of blocks;
- how the Lanczos block works with products only, and how many steps it needs.

**Before you start:** [tutorial 1](01-install-and-first-check.md). [Tutorial 2](02-write-your-own-svd.md)
explains the `svd(a, rank)` contract used here.

**Notebook version:** [notebooks/03_blocks_and_recipes.ipynb](../../notebooks/03_blocks_and_recipes.ipynb)
(with a cell for your own QR).

## 3.1 Two layers: blocks and recipes

`svdlab` exposes both levels:

- **`svdlab.blocks`**: the individual steps, one matrix at a time. They are this
  repository's own implementations built from tensor arithmetic and matrix products. You
  can read them, and they run unchanged on CPU and GPU, real and complex.
- **`svdlab.recipes`**: complete SVDs assembled from blocks. Every recipe is itself a
  method (it has the `svd(a, rank)` signature), and every step is a keyword argument, so
  you can replace any one step with your own function.

```text
recipes.golub_kahan   a ──bidiagonalize──► Q, B, P ──bidiagonal_svd──► U_B, s, V_B ──► U = Q U_B, V = P V_B
recipes.qr_then_svd   a ──qr──► Q, R ──small_svd(R)──► U_R, s, Vh ──► U = Q U_R
recipes.randomized    a ──sketch──► A Ω ──qr──► Q ──(power iterations: qr)──► Q ──small_svd(Qᴴ A)──► U = Q U_small
recipes.lanczos       a ──lanczos_bidiagonalize (k steps)──► U, B, V ──bidiagonal_svd──► U U_B, V V_B
recipes.jacobi        a ──jacobi_svd──► U, s, Vh
```

What the recipes do, in words:

| Recipe | Idea | Good for | Replaceable steps |
| --- | --- | --- | --- |
| `golub_kahan` | The textbook dense SVD (LAPACK's `gesvd` works this way): Householder reflections reduce A to a bidiagonal matrix B, then an iterative QR method diagonalizes B | full SVDs of dense matrices | `bidiagonalize`, `bidiagonal_svd` |
| `qr_then_svd` | First A = Q R. R is only n × n, so its SVD is cheaper; then U = Q U_R | tall matrices (m much larger than n) | `qr`, `small_svd` |
| `randomized` | Multiply A by a random matrix Ω to find a basis Q of its dominant columns, sharpen it with power iterations, then take the SVD of the small matrix Qᴴ A | the top few triplets of a large matrix whose singular values decay | `sketch`, `qr`, `small_svd`, and the parameters `oversample`, `power_iterations`, `seed` |
| `lanczos` | Build B one column at a time using only products A x and Aᴴ y; the largest singular values of B converge to those of A first | the top triplets, and matrices you only have as a product (sparse, implicit) | `bidiagonalize` (the Lanczos block), `bidiagonal_svd`, and `steps`, `seed` |
| `jacobi` | Rotate pairs of columns until all are orthogonal | small matrices, highest accuracy, a reference | none |

## 3.2 The blocks

| Block | Input | Returns |
| --- | --- | --- |
| `qr(a)` | tall `(m, n)` | `Q (m, n)` orthonormal, `R (n, n)` upper triangular (Householder) |
| `torch_qr(a)` | tall `(m, n)` | the same from the library (LAPACK / cuSOLVER), as a control |
| `bidiagonalize(a)` | tall `(m, n)` | `Q (m, n)`, `d (n,)`, `e (n-1,)`, `P (n, n)` with `a = Q B Pᴴ`; `B` has the real diagonal `d` and superdiagonal `e` |
| `lanczos_bidiagonalize(a, steps, seed=0)` | tall `(m, n)`, `steps <= n` | `U (m, k)`, `d (k,)`, `e (k-1,)`, `V (n, k)` with `Uᴴ a V = B`, k = steps; uses `a` only through products, the top singular values of B converge first |
| `bidiagonal_svd(d, e)` | real `d`, `e` | `U, s, V` with `B = U diag(s) Vᵀ` (implicit-shift QR, LAPACK's bdsqr). Note: `V`, not `Vh` |
| `jacobi_svd(a, rank=None)` | any `(m, n)` | `U, s, Vh` (one-sided Jacobi) |
| `tridiagonalize(h)` | Hermitian `(n, n)` | `Q, d, e` with `h = Q T Qᴴ` |
| `hermitian_eig(h)` | Hermitian `(n, n)` | `w` ascending, `V` (Jacobi eigensolver) |
| `cholesky(g)` | Hermitian positive definite | lower `L` with `g = L Lᴴ` |
| `solve_lower(l, b)`, `solve_upper(u, b)` | triangular, right-hand sides | `X` |
| `random_sketch(a, columns, seed=0)` | `(m, n)` | Gaussian `(n, columns)` with `a`'s dtype and device |
| `complete_basis(q, size)` | orthonormal `(m, k)` | `size` orthonormal columns |
| `svd(a, algorithm, rank=None)` | any | any of the 16 repository SVDs (`blocks.ALGORITHMS`) |
| `torch_svd(a, rank=None)` | any | library SVD, as a control |

Blocks marked *tall* raise a `ValueError` on a wide matrix. The recipes factorize the
adjoint of a wide matrix, so the steps you plug into them only ever see tall input.

You can call the blocks directly to see what they produce:

```python
import torch
from svdlab import blocks

a = torch.randn(8, 5, dtype=torch.float64)
q, d, e, p = blocks.bidiagonalize(a)
b = torch.diag(d) + torch.diag(e, 1)                 # the 5x5 upper bidiagonal matrix
print(torch.linalg.norm(q @ b @ p.mH - a))           # about 3e-15
u, s, v = blocks.bidiagonal_svd(d, e)
print(s)                                             # the singular values of a
```

## 3.3 Replace a step

A replacement must keep the contract of the block it replaces: same inputs, same
outputs, same shapes. Here is CholeskyQR2, a QR built from a Cholesky factorization of
YᴴY. It is popular on GPUs because it is two matrix products and two small triangular
operations. It is in [examples/replace_a_step.py](../../examples/replace_a_step.py):

```python
import torch
from svdlab import recipes

def cholesky_qr2(y):
    """Same contract as svdlab.blocks.qr: tall y (m, n) -> Q (m, n) orthonormal, R (n, n) upper."""
    q, r = y, torch.eye(y.shape[1], dtype=y.dtype, device=y.device)
    for _ in range(2):                            # the second pass repairs the lost orthogonality
        l = torch.linalg.cholesky(q.mH @ q)       # QᴴQ = L Lᴴ
        q = torch.linalg.solve_triangular(l.mH, q, upper=True, left=False)   # Q L⁻ᴴ
        r = l.mH @ r
    return q, r

def svd(a, rank):
    return recipes.randomized(a, rank, qr=cholesky_qr2)
```

Check it next to the unchanged recipe:

```bash
python -m svdlab check examples/replace_a_step.py --compare svdlab.recipes:randomized
```

```text
replace_a_step       ill_conditioned_1e8    ERROR _LinAlgError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite (...)
...
replace_a_step: 7/8 ok, total median 3935.889 ms, failed: ill_conditioned_1e8
recipes.randomized: 8/8 ok, total median 7062.109 ms
```

The replacement is almost twice as fast and fails exactly where theory says it should:
YᴴY squares the condition number (the same effect as in tutorial 2), and for a condition
number of 1e8 the Gram matrix is no longer positive definite in float64. Whether that is
acceptable depends on your matrices; the check makes the trade-off visible instead of
hiding it in an average. (The float32 low-rank case is close to the limit as well and can
fail on some hardware.)

`--compare svdlab.recipes:randomized` shows the method form `module:function`: any
importable function with the `(a, rank)` signature can be checked or compared.

## 3.4 Find the slow step

Replacing a block with the library version is also a quick profiler. The repository's
own steps are written for readability and portability, not speed. On the quick matrices
(CPU, times from one laptop):

| Method | Steps | Cases ok | Total median |
| --- | --- | --- | --- |
| `recipes.golub_kahan` | `bidiagonalize`, `bidiagonal_svd` | 6/6 | 3412 ms |
| the same, `bidiagonal_svd` from the library | `bidiagonalize`, `torch.linalg.svd(B)` | 6/6 | 59 ms |
| `recipes.qr_then_svd` | `qr`, `jacobi_svd` | 6/6 | 569 ms |
| the same, both steps from the library | `torch_qr`, `torch_svd` | 6/6 | 2.7 ms |

So in Golub–Kahan almost all of the time is spent in the bidiagonal SVD, which is where a
faster implementation would pay off. To try a library step in a recipe:

```python
import functools, torch
from svdlab import blocks, recipes, compare, summary, QUICK_SUITE

def library_bidiagonal_svd(d, e):
    """Same contract as blocks.bidiagonal_svd: B = U diag(s) Vᵀ, returns (U, s, V)."""
    u, s, vh = torch.linalg.svd(torch.diag(d) + torch.diag(e, 1))
    return u, s, vh.mT

methods = {
    'golub_kahan': recipes.golub_kahan,
    'gk_library_bdsvd': functools.partial(recipes.golub_kahan, bidiagonal_svd=library_bidiagonal_svd),
    'qr_then_svd': recipes.qr_then_svd,
    'qr_then_svd_library': functools.partial(recipes.qr_then_svd, qr=blocks.torch_qr, small_svd=blocks.torch_svd),
}
rows = compare(methods, QUICK_SUITE, repeat=3, verbose=False)
for s in summary(rows):
    print(s['method'], f"{s['ok']}/{s['cases']}", f"{s['total_median_ms']:.1f} ms")
```

`functools.partial(recipe, step=...)` turns a recipe with one replaced step into an
`(a, rank)` function without writing a wrapper.

## 3.5 Tune a parameter

`recipes.randomized` also takes numerical parameters: `oversample` (extra sketch
columns, default 8), `power_iterations` (default 2) and `seed`. A truncated case passes
when its residual is within 1.05 times the best possible one (`tail×`, see tutorial 5).
Two rank-10 problems, a matrix of exact rank 10 plus noise and a matrix whose singular
values decay slowly (1 to 1e-3 geometrically):

| Variant | `low_rank_k10` tail× | `slow_decay` tail× | verdict |
| --- | --- | --- | --- |
| `power_iterations=0` | 1.38 | 1.10 | both FAIL |
| `power_iterations=0, oversample=30` | 1.12 | 1.03 | one FAIL, 3x slower |
| `power_iterations=2` (default) | 1.00 | 1.00 | both ok |

```python
cases = [
    dict(name='low_rank_k10', m=300, n=200, family='low_rank', rank=10, tail=1e-6, output_rank=10),
    dict(name='slow_decay', m=300, n=200, family='geometric', condition=1e3, output_rank=10),
]
compare({'p0': functools.partial(recipes.randomized, power_iterations=0),
         'p0_os30': functools.partial(recipes.randomized, power_iterations=0, oversample=30),
         'p2': recipes.randomized}, cases, repeat=3)
```

## 3.6 Compose recipes

A recipe is itself an SVD, so it can be a step of another recipe. QR first, then
Golub–Kahan on the small triangular factor:

```python
qr_then_golub_kahan = functools.partial(recipes.qr_then_svd, small_svd=recipes.golub_kahan)
```

And you can write your own chain from blocks in a few lines, in a file that
`python -m svdlab check` accepts:

```python
import torch
from svdlab import blocks

def svd(a, rank):
    wide = a.shape[0] < a.shape[1]
    x = a.mH if wide else a
    q, r = blocks.qr(x)                       # x = Q R
    q2, d, e, p = blocks.bidiagonalize(r)     # R = Q2 B Pᴴ
    ub, s, vb = blocks.bidiagonal_svd(d, e)   # B = Ub diag(s) Vbᵀ
    u = q @ q2 @ ub.to(x.dtype)
    v = p @ vb.to(x.dtype)
    u, s, vh = u[:, :rank], s[:rank], v[:, :rank].mH
    return (vh.mH, s, u.mH) if wide else (u, s, vh)
```

(`bidiagonal_svd` returns real `U` and `V`; `.to(x.dtype)` makes them complex when the
input is complex.)

## 3.7 Lanczos: a step that only needs products with A

`blocks.lanczos_bidiagonalize(a, steps)` builds the bidiagonal matrix one column at a
time, and it touches `a` only through the products `a @ x` and `aᴴ @ y`. Its largest
singular values converge first, so a few steps give the top of the spectrum.
`recipes.lanczos` runs it and then solves the small bidiagonal problem with
`bidiagonal_svd`, which you can replace. [examples/lanczos_then_bdsqr.py](../../examples/lanczos_then_bdsqr.py)
plugs in the library SVD for that last step:

```python
import torch
from svdlab import recipes

def my_bdsqr(d, e):
    """Same contract as svdlab.blocks.bidiagonal_svd: B = U diag(s) Vᵀ, s descending."""
    u, s, vh = torch.linalg.svd(torch.diag(d) + torch.diag(e, 1))
    return u, s, vh.mT

def svd(a, rank):
    return recipes.lanczos(a, rank, bidiagonal_svd=my_bdsqr)
```

```bash
python -m svdlab check examples/lanczos_then_bdsqr.py --quick --compare svdlab.recipes:lanczos
```

```text
lanczos_then_bdsqr: 6/6 ok, total median 22.354 ms
recipes.lanczos: 6/6 ok, total median 2058.092 ms
```

The number of steps (default `min(m, n, 2·rank + 8)`) is the accuracy knob. On the two
rank-10 problems of section 3.5:

| `steps` | `low_rank_k10` | `slow_decay` |
| --- | --- | --- |
| 12 | ok, tail× 1.00 | **FAIL**, tail× 1.13, a top value up to 61 % off |
| 28 (default) | ok, tail× 1.00 | ok, tail× 1.00, σ-rel 4e-5 |
| 60 | ok, tail× 1.00 | ok, tail× 1.00, σ-rel 3e-15 |

A clear gap after σ₁₀ (the low-rank matrix) needs very few steps. A slowly decaying
spectrum needs more, and in exchange it costs more time.

## 3.8 The 16 repository SVDs

`blocks.svd(a, algorithm)` and the builtin method names give access to the complete SVDs
in `svdbench/standalone`: `bidiag_qr, bidiag_dc, jacobi, block_jacobi, qr, tsqr,
gram_jacobi, gram_dc, polar_qdwh, polar_newton_schulz, polar_halley, polar_polynomial,
randomized, subspace, block_krylov, lanczos`. They are good references to compare a new
method with:

```bash
python -m svdlab check my_svd.py --quick --compare torch,bidiag_qr,polar_qdwh
```

## Next

- [Tutorial 4](04-native-cpp-cuda.md): write the step, or the whole SVD, in C++ or CUDA.
- [Tutorial 5](05-metrics-and-matrices.md): build test matrices that stress your step.
