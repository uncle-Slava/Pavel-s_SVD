# Tutorial 1: Install and run your first check

**What this tutorial is about.** You set up a Python environment, install `svdlab`, and run
a complete SVD that ships with the repository through the checker. Then you go through
its report line by line. You don't write any code yet. The point is to see what the tool
does and to make sure everything works on your machine.

**You will learn:**

- how to install PyTorch and `svdlab` (CPU only, or with a GPU);
- what happens when you run `python -m svdlab check`;
- how to read a report line: verdict, time, and the four error numbers;
- why some correct results still show large errors;
- how to compare with reference methods, run on a GPU and use the exit code in scripts.

**Before you start:** if the words *method*, *case*, *verdict* or *reference* are new, read
[Start here](00-start-here.md) first (10 minutes).

**Notebook version:** [notebooks/01_quickstart.ipynb](../../notebooks/01_quickstart.ipynb)
runs the same steps from Python.

## 1.1 What svdlab does

You give `svdlab` a function that computes an SVD. It runs that function on a fixed set of
test matrices (well conditioned, ill conditioned, clustered singular values, complex,
low rank, ...), compares every result with an independent double-precision reference and
tells you, per matrix:

- whether the answer is correct (`ok` or `FAIL`),
- how large the errors are (reconstruction, orthogonality, singular values),
- how long one call takes (median of several runs).

You can call it from a terminal (`python -m svdlab check ...`) or from Python
(`svdlab.check(...)`). Both give the same numbers.

## 1.2 Requirements

- Python 3.10 or newer.
- PyTorch 2.0 or newer. Install the build that matches your hardware **before** installing
  this package, following <https://pytorch.org/get-started/locally/>:
  - a laptop without an NVIDIA GPU: the CPU build is enough for every tutorial;
  - an NVIDIA GPU: the CUDA build;
  - an AMD GPU: the ROCm build. Do not install a CUDA build over a working ROCm one.
- Git.

For C++ and CUDA (tutorial 4 only) you also need a C++ compiler, the Python headers and,
for CUDA, the CUDA toolkit. You can install those later.

## 1.3 Install

```bash
git clone https://github.com/upv/Pavel-s_SVD.git
cd Pavel-s_SVD
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install torch            # or the command from pytorch.org for your GPU
python -m pip install -e '.[laboratory]'
```

`-e` installs the repository in editable mode: changes you make to files in it are used
immediately, without reinstalling. The `laboratory` extra adds JupyterLab and pandas for
the notebooks.

Check the installation:

```bash
python -c "import torch, svdlab; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu only')"
```

## 1.4 Your first check

The repository ships a working SVD to start from, [examples/my_svd.py](../../examples/my_svd.py)
(a one-sided Jacobi SVD written in plain PyTorch). Check it on the small matrices:

```bash
python -m svdlab check examples/my_svd.py --quick
```

The output looks like this (times depend on your machine):

```text
recon  = |A - U diag(s) Vh| / |A|          tail× = recon / best possible (truncated cases)
σ-rel  = max relative error of the values   orth  = max(|UᴴU - I|, |V Vᴴ - I|) / sqrt(k)
ok     = orth and recon within 100·eps·max(m, n), or tail× <= 1.05

my_svd               gaussian_tall          ok      64.858 ms  recon 4.51e-15  tail× -  σ-rel 3.64e-15  orth 4.66e-15
my_svd               gaussian_wide          ok      44.770 ms  recon 4.34e-15  tail× -  σ-rel 3.71e-15  orth 4.25e-15
my_svd               ill_conditioned_1e8    ok      92.508 ms  recon 6.48e-15  tail× -  σ-rel 4.24e-10  orth 6.60e-15
my_svd               clustered              ok      79.093 ms  recon 8.67e-15  tail× -  σ-rel 4.75e-14  orth 8.29e-15
my_svd               complex_gaussian       ok      81.754 ms  recon 3.04e-15  tail× -  σ-rel 2.44e-15  orth 3.20e-15
my_svd               low_rank_k4            ok     195.062 ms  recon 6.64e-06  tail× 1.00e+00  σ-rel 3.99e-15  orth 7.96e-15

my_svd: 6/6 ok, total median 558.045 ms
```

## 1.5 Reading one line

Take the first result line:

```text
my_svd   gaussian_tall   ok   64.858 ms   recon 4.51e-15   tail× -   σ-rel 3.64e-15   orth 4.66e-15
```

| Field | Meaning here |
| --- | --- |
| `my_svd` | The method: the file name without `.py`. |
| `gaussian_tall` | The test matrix: 48×32, random Gaussian entries, float64. `python -m svdlab list` shows all matrices. |
| `ok` | The answer is correct within the tolerance for float64 at this size. |
| `64.858 ms` | Median time of one call. Only your function is timed, not the checking. |
| `recon` | ‖A − U diag(s) Vᴴ‖ / ‖A‖: how well the factors rebuild the matrix. About 1e-15 is the best float64 can do. |
| `tail×` | Only for truncated (low-rank) cases, so `-` here. |
| `σ-rel` | Largest relative error of a singular value. |
| `orth` | How far U and V are from having orthonormal columns. |

Two lines deserve a second look:

- `ill_conditioned_1e8`: `σ-rel` is 4e-10, much larger than on the other matrices, and the
  case is still `ok`. The smallest singular values of this matrix are 1e8 times smaller
  than the largest, and nobody can compute them to full relative accuracy in float64.
  The verdict looks at errors relative to the whole matrix, which are still about 1e-15.
- `low_rank_k4`: the method is asked for only 4 singular triplets. `recon` is 6.6e-6
  because the rest of the matrix is deliberately left out. `tail× 1.00` says this is
  exactly as good as the best possible rank-4 approximation.

[Tutorial 5](05-metrics-and-matrices.md) explains every metric in detail.

## 1.6 Compare with other methods

`--compare` runs reference methods on the same matrices. They are shown next to yours
and never change the exit code:

```bash
python -m svdlab check examples/my_svd.py --quick --compare torch,jacobi
```

`torch` is `torch.linalg.svd` (LAPACK on the CPU, cuSOLVER on the GPU). `jacobi` is the
repository's own Jacobi SVD. `python -m svdlab list` prints all builtin methods.

Without `--quick` the check uses 8 larger matrices (up to 400×300). That is the set to
report results on; `--quick` is for a fast first look.

## 1.7 On the GPU

If `torch.cuda.is_available()` printed `True` above:

```bash
python -m svdlab check examples/my_svd.py --quick --device cuda
python -m svdlab check examples/my_svd.py --quick --gpu-check
```

The first command runs everything on the GPU (ROCm builds use the same `cuda` spelling).
The second runs on the CPU and on the GPU and passes a case only when both are correct,
the results really come back on the GPU and the singular values agree. On a CPU-only
machine both commands stop with an error saying that torch sees no CUDA device.

## 1.8 Exit codes

`python -m svdlab check` returns 0 when every case of every checked method is `ok`, 1 when
a case fails and 2 when a method cannot be loaded or compiled. You can use it as is in CI
or in a shell script:

```bash
python -m svdlab check my_svd.py --quick || echo "my_svd has a failing case"
```

## Next

- [Tutorial 2](02-write-your-own-svd.md): write your own SVD.
- [Tutorial 3](03-blocks-and-recipes.md): replace one step of an existing SVD.
