# Quickstart notebooks

Run them in order; each one matches a [tutorial](../docs/tutorials/README.md). If you are new,
read [Start here](../docs/tutorials/00-start-here.md) first: what svdlab does, the words the
notebooks use, and which parts of the repository you need.

Each notebook starts with what it is about, what you will learn and the words it uses.
Before every code cell it says what the cell does and what you should see.

| Notebook | What you do | CPU time |
| --- | --- | --- |
| [01_quickstart.ipynb](01_quickstart.ipynb) | run the starter SVD, read the report, compare with torch | about 1 min |
| [02_write_your_own_svd.ipynb](02_write_your_own_svd.ipynb) | write an SVD, see why it fails on hard matrices, fix it, save it to a file | about 1 min |
| [03_blocks_and_recipes.ipynb](03_blocks_and_recipes.ipynb) | replace one step (QR, bidiagonal SVD) of a ready SVD, find the slow step, tune parameters | about 2 min |
| [04_native_cpp_cuda.ipynb](04_native_cpp_cuda.ipynb) | compile and check C++ and CUDA files | about 2 min (first builds) |
| [05_metrics_and_matrices.ipynb](05_metrics_and_matrices.ipynb) | every metric, the verdict, your own matrices, CSV output, the GPU check | about 1 min |

```bash
python -m pip install -e '.[laboratory]'      # once, from the repository root
python -m jupyterlab notebooks/
```

Everything runs on a CPU. When PyTorch sees a CUDA or ROCm GPU, the notebooks use it
(set `USE_GPU = False` in the first cell to stay on the CPU) and run the GPU-only parts.
Cells that need a C++ compiler or the CUDA toolkit print what is missing and skip.

The saved outputs come from a CPU run; your times will differ. Files the notebooks write
go to `laboratory_runs/`, which git ignores.
