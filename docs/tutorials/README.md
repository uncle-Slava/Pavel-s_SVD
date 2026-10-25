# svdlab tutorials

These tutorials are for developers who want to write a singular value decomposition, or
change one step of an existing one, and see how accurate and how fast it is. You do not
need to know anything about the rest of this repository (the benchmark harness, the
playground, the research reports). Everything goes through one package, `svdlab`.

Each tutorial has a matching notebook in [`notebooks/`](../../notebooks/) that you can run
top to bottom. The notebooks run on a CPU-only laptop; on a machine with an NVIDIA (CUDA)
or AMD (ROCm) GPU they also run the GPU parts.

| # | Tutorial | Notebook | Time | You will |
| --- | --- | --- | --- | --- |
| 1 | [Install and run your first check](01-install-and-first-check.md) | [01_quickstart.ipynb](../../notebooks/01_quickstart.ipynb) | 10 min | install, run the starter SVD, read one result line |
| 2 | [Write your own SVD](02-write-your-own-svd.md) | [02_write_your_own_svd.ipynb](../../notebooks/02_write_your_own_svd.ipynb) | 20 min | learn the function contract, write an SVD, debug a failing one |
| 3 | [Build an SVD from blocks and recipes](03-blocks-and-recipes.md) | [03_blocks_and_recipes.ipynb](../../notebooks/03_blocks_and_recipes.ipynb) | 20 min | replace one step (QR, bidiagonal SVD, ...) of a ready algorithm |
| 4 | [C++ and CUDA](04-native-cpp-cuda.md) | [04_native_cpp_cuda.ipynb](../../notebooks/04_native_cpp_cuda.ipynb) | 15 min | compile one `.cpp` or `.cu` file and check it like Python code |
| 5 | [Read the metrics and test your own matrices](05-metrics-and-matrices.md) | [05_metrics_and_matrices.ipynb](../../notebooks/05_metrics_and_matrices.ipynb) | 20 min | understand every number, use your own matrices, save results, check the GPU |

Start with tutorial 1. After that the order is up to you: 2 if you write a whole
algorithm, 3 if you improve one step, 4 if you write C++ or CUDA. Tutorial 5 is the
reference for reading results.

## Running the notebooks

```bash
python -m pip install -e '.[laboratory]'     # adds JupyterLab and pandas
python -m jupyterlab notebooks/
```

Open a notebook and choose *Run → Run All Cells*. Each notebook finds the repository by
itself, so it also works when opened from VS Code or PyCharm, as long as the selected
Python environment is the one where you installed the package.

## Where to go next

- [Developer guide](../DEVELOPER_GUIDE.md): the compact reference (every option, every
  block, exit codes, troubleshooting).
- [Algorithm laboratory](../ALGORITHM_LABORATORY.md): trace what happens inside an
  algorithm, iteration by iteration.
