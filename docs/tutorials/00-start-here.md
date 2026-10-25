# Start here: what svdlab is, and what is what

Read this page first. It takes about ten minutes. It explains what the tools do, which
words the other tutorials use, and which parts of this large repository you need. The
other parts you can safely ignore.

## Why this exists

Say you have an idea for computing a singular value decomposition: a new algorithm, a
faster QR inside an existing one, a CUDA kernel for one step. Before anyone can use it,
you need answers to three questions:

1. **Is it correct?** Not just on a random matrix, but also on the matrices that break
   SVD algorithms: ill-conditioned ones, nearly equal singular values, complex numbers,
   wide shapes, low precision.
2. **How fast is it?** Measured fairly, against the same matrices and the same reference
   methods as everybody else.
3. **Does it really run on the GPU?** And does it give the same answers there as on the
   CPU?

`svdlab` answers all three with one command. You write one function. It runs that
function on a fixed set of test matrices, compares every result with a trusted reference
and prints a verdict per matrix. You don't have to learn the benchmark harness, the
playground or the research reports that make up most of this repository.

## A two-minute SVD refresher

Every real or complex `m × n` matrix A can be written as

```text
A = U · diag(s) · Vᴴ
```

| Factor | Shape (for k = min(m, n)) | Property |
| --- | --- | --- |
| `U` | `m × k` | columns are orthonormal: UᴴU = I (the *left singular vectors*) |
| `s` | `k` | real, non-negative, sorted from largest to smallest (the *singular values* σ₁ ≥ σ₂ ≥ … ≥ 0) |
| `Vh` = Vᴴ | `k × n` | rows are orthonormal: Vh Vhᴴ = I (Vᴴ, the conjugate transpose of the *right singular vectors*) |

`ᴴ` means conjugate transpose. For real matrices it is the plain transpose ᵀ. In
PyTorch it is `x.mH`.

- **Full SVD:** all k = min(m, n) triplets (σᵢ, uᵢ, vᵢ).
- **Truncated (low-rank) SVD:** only the largest `rank` triplets. They give the best
  possible rank-`rank` approximation of A (the Eckart–Young theorem). This is what
  compression, PCA and many solvers need.
- **Condition number:** σ₁ / σ_min. When it is large (1e8, 1e12), the small singular
  values are very hard to compute accurately, and many fast shortcuts break.

Why checking an SVD is harder than it looks:

- The product U diag(s) Vᴴ can rebuild A perfectly while U is far from orthonormal.
  Tutorial 2 shows a real example. So `svdlab` checks orthogonality separately.
- Singular vectors are not unique: (u, v) and (−u, −v) are both correct. Inside a group
  of equal singular values, any orthonormal basis of the group is correct. So you can't
  compare vectors entry by entry with another library.
- Floating point has limits: float64 keeps about 16 digits, float32 about 7. The
  tolerances have to scale with the precision and the matrix size.

## How svdlab works

```text
 your function            test matrices               reference
 svd(a, rank)  ◄── a ───  (fixed seeds, same   ──►   LAPACK SVD in float64
      │                    bytes everywhere)          of the exact same input
      │ U, s, Vh                                             │
      ▼                                                      ▼
 ┌──────────────────────── metrics ─────────────────────────────┐
 │ reconstruction · orthogonality · singular value errors · ... │
 └──────────────────────────────┬───────────────────────────────┘
                                ▼
            verdict per matrix: ok / FAIL / ERROR   +   median time
```

1. `svdlab` creates each test matrix from a fixed seed, so every run on every machine
   sees exactly the same numbers.
2. It computes a reference SVD of that matrix with LAPACK in float64.
3. It calls your function once and compares the result with the reference.
4. It calls your function a few more times, untimed and then timed, and reports the
   median time of the call alone.
5. It prints one line per matrix and a summary, and returns an exit code (0 = all ok).

## Words used in the tutorials

| Word | Meaning | Where you meet it |
| --- | --- | --- |
| **method** | Something that computes an SVD: a Python function, a `.py` file with a function `svd`, a `.cpp`/`.cu` file, or a builtin name such as `torch` | everywhere |
| **contract** | The rule every method follows: `svd(a, rank)` returns `U, s, Vh` with the shapes above | tutorial 2 |
| **case** | One test matrix plus the rank asked for, for example `ill_conditioned_1e8` (160×120, condition number 1e8) | every report line |
| **suite** | A fixed list of cases. `DEFAULT_SUITE`: 8 matrices up to 400×300, used to report results. `QUICK_SUITE` (`--quick`): 6 smaller ones for a first try | tutorial 1 |
| **reference** | The float64 LAPACK SVD of the exact input that your result is compared with | tutorial 5 |
| **metric** | One number that measures one kind of error, for example `recon` or `orth` | tutorial 5 |
| **tolerance** | How large an error may be: `100 · eps · max(m, n)`, where `eps` is the precision of the input's dtype | tutorial 5 |
| **verdict** | `ok` (correct within tolerance), `FAIL` (computed, but wrong) or `ERROR` (the method raised an exception) | every report line |
| **block** | One step of an SVD algorithm, such as a QR factorization or a bidiagonalization (`svdlab.blocks`) | tutorial 3 |
| **recipe** | A complete SVD built from blocks. Every block in it can be replaced (`svdlab.recipes`) | tutorial 3 |
| **builtin method** | A ready-made method you can name instead of writing one: `torch`, `numpy`, `jacobi`, `bidiag_qr`, … (`python -m svdlab list`) | `--compare` |
| **native method** | A method written in C++ or CUDA and compiled automatically | tutorial 4 |
| **device** | Where the computation runs: `cpu` or `cuda` (an NVIDIA or AMD GPU) | `--device` |
| **GPU check** | Runs a method on CPU and GPU and checks that both are correct, that they agree, and that the GPU result really stays on the GPU | tutorial 5 |
| **rows** | What `check` and `compare` return in Python: one dictionary per case with every metric | tutorial 5 |

## What is what in the repository

You need only the first four rows. The rest is the research project this tool came out
of.

| Path | What it is | Do you need it? |
| --- | --- | --- |
| `svdlab/` | The tool itself: the command line, `check`, `blocks`, `recipes`, `native` | **yes**, you import from it |
| `examples/` | Small files to copy and edit: a starter SVD, a replaced step, C++ and CUDA templates | **yes**, start from these |
| `docs/tutorials/`, `notebooks/` | These tutorials and the notebooks that go with them | **yes** |
| `docs/DEVELOPER_GUIDE.md` | One-page reference: every option, every block, troubleshooting | **yes**, once you know the basics |
| `tests/` | Unit tests (`python -m unittest discover -s tests`) | only if you change `svdlab` |
| `svdbench/standalone/` | The 16 SVD implementations and the primitives that `svdlab.blocks` exposes | only to read how a block works |
| `algorithm_laboratory/` | Tools to trace the inside of an algorithm iteration by iteration | later, see [the laboratory](../ALGORITHM_LABORATORY.md) |
| `svdbench/` (rest), `configs/`, `corpus/`, `results/`, `tools/`, `references/`, `deploy/` | Benchmark harness, saved measurements, reports, deployment of the web playground | no |
| `docs/` (other files) | Research reports and benchmark results | no |

## Three ways to use it

| Way | Looks like | Best for |
| --- | --- | --- |
| **Command line** | `python -m svdlab check my_svd.py --quick` | the everyday loop: edit a file, run, read the report. Also CI |
| **Python** | `rows = svdlab.check(my_svd)` | scripts, your own matrices, analyzing results with pandas |
| **Notebooks** | `notebooks/01_quickstart.ipynb` | learning, experimenting, plots |

All three run the same code and give the same numbers.

## svdlab at a glance

| Name | What it does |
| --- | --- |
| `python -m svdlab check METHOD [...]` | Check one or more methods; options: `--quick`, `--compare`, `--device`, `--gpu-check`, `--case`, `--csv`, … |
| `python -m svdlab list` | Print the builtin methods, blocks, recipes and test matrices |
| `svdlab.check(method, cases)` | The same check from Python; returns rows |
| `svdlab.compare({name: method, ...}, cases)` | Several methods on the same cases |
| `svdlab.gpu_check(method, cases)` | CPU against GPU |
| `svdlab.table(rows)`, `svdlab.summary(rows)`, `svdlab.save(rows, path)` | Rows as a pandas table, per-method summary, CSV/JSON file |
| `svdlab.QUICK_SUITE`, `svdlab.DEFAULT_SUITE` | The standard cases |
| `svdlab.MatrixCase` | Build your own test matrix: by family, from chosen singular values, or from a `.npy` file |
| `svdlab.load_file(path)` | Load the method from a `.py`, `.cpp` or `.cu` file |
| `svdlab.blocks`, `svdlab.recipes` | Steps of SVD algorithms and complete SVDs built from them |
| `svdlab.native` | Compile a C++/CUDA file; check which build tools are missing |

## What you need to know beforehand

- **Python:** functions, imports, running a script.
- **PyTorch basics:** tensors, `@` for matrix products, `torch.linalg`. If you know
  NumPy, PyTorch reads almost the same.
- **Linear algebra:** what an SVD and a QR factorization are. You don't need to know how
  they are computed. The tutorials explain the numerical effects as they come up.
- C++ and CUDA only for tutorial 4.

## Where to go next

[Tutorial 1: install and run your first check](01-install-and-first-check.md), or open
[notebooks/01_quickstart.ipynb](../../notebooks/01_quickstart.ipynb).
