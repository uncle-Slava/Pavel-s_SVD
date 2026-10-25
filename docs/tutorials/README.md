# svdlab tutorials

These tutorials are for developers who want to write a singular value decomposition (SVD),
or improve one step of an existing one, and find out whether it is correct and how fast
it is, on a CPU or a GPU.

You don't need to know anything about the rest of this repository. It is a large
research project (benchmark harness, web playground, reports), and almost none of it is
needed here. Everything you use goes through one small package, `svdlab`. Which parts of
the repository matter is explained in [Start here](00-start-here.md).

## What you will be able to do

- Run a ready-made SVD through the checker and read its report (tutorial 1).
- Write your own SVD as one Python function, see where it fails and why, and fix it
  (tutorial 2).
- Take a complete algorithm and replace a single step, such as the QR factorization,
  with your own version (tutorial 3).
- Do the same in C++ or with your own CUDA kernel, without CMake (tutorial 4).
- Understand every number in the report, test on your own matrices and save results
  (tutorial 5).

## The path

Every tutorial has a matching notebook in [`notebooks/`](../../notebooks/) that you can run
top to bottom. Read the text, run the notebook, or both. The notebooks work on a laptop
without a GPU. When PyTorch sees an NVIDIA (CUDA) or AMD (ROCm) GPU, they use it.

| # | Tutorial | Notebook | Time | You will |
| --- | --- | --- | --- | --- |
| 0 | [Start here: what is what](00-start-here.md) | | 10 min | learn the idea, an SVD refresher, the words used, the repository map |
| 1 | [Install and run your first check](01-install-and-first-check.md) | [01_quickstart.ipynb](../../notebooks/01_quickstart.ipynb) | 10 min | install, run the starter SVD, read the report |
| 2 | [Write your own SVD](02-write-your-own-svd.md) | [02_write_your_own_svd.ipynb](../../notebooks/02_write_your_own_svd.ipynb) | 20 min | learn the function contract, write an SVD, debug a failing one |
| 3 | [Build an SVD from blocks and recipes](03-blocks-and-recipes.md) | [03_blocks_and_recipes.ipynb](../../notebooks/03_blocks_and_recipes.ipynb) | 20 min | replace one step (QR, bidiagonal SVD, ...) of a ready algorithm |
| 4 | [C++ and CUDA](04-native-cpp-cuda.md) | [04_native_cpp_cuda.ipynb](../../notebooks/04_native_cpp_cuda.ipynb) | 15 min | compile one `.cpp` or `.cu` file and check it like Python code |
| 5 | [Read the metrics and test your own matrices](05-metrics-and-matrices.md) | [05_metrics_and_matrices.ipynb](../../notebooks/05_metrics_and_matrices.ipynb) | 20 min | understand every number, use your own matrices, save results, check the GPU |

Read 0 and 1 first. After that, choose by what you want to do: 2 to write a whole
algorithm, 3 to improve one step, 4 for C++ or CUDA. Tutorial 5 is the reference for
reading results; come back to it whenever a number is unclear.

## What you need

- Python 3.10 or newer and PyTorch 2.0 or newer (tutorial 1 shows how to install).
- Basic Python and PyTorch, and knowing what an SVD and a QR factorization are. How they
  are computed is not needed; the tutorials explain the numerical effects as they come up.
- For tutorial 4 only: a C++ compiler, and for CUDA the CUDA toolkit.

## Running the notebooks

```bash
python -m pip install -e '.[laboratory]'     # adds JupyterLab and pandas
python -m jupyterlab notebooks/
```

Open a notebook and choose *Run → Run All Cells*, or run the cells one by one with
Shift+Enter, reading the text between them. Each notebook finds the repository by itself,
so it also works from VS Code or PyCharm, as long as the selected Python environment (the
"kernel") is the one where you installed the package.

## Where to go next

- [Developer guide](../DEVELOPER_GUIDE.md): the compact reference (every option, every
  block, exit codes, troubleshooting).
- [Algorithm laboratory](../ALGORITHM_LABORATORY.md): trace what happens inside an
  algorithm, iteration by iteration.
