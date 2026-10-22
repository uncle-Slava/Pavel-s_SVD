# Algorithm laboratory

A Python workbench for changing numerical algorithms and inspecting their steps.
It runs independently of the web playground.

**Just score your own SVD.** Write `def my_svd(a, rank): return U, s, Vh` and open
[my_svd.ipynb](my_svd.ipynb), or from the repository root:

```bash
python -m algorithm_laboratory.scorecard --method my_module:my_svd --method torch
```

Each method runs on the same 8 standard matrices and gets reconstruction, orthogonality,
spectrum and subspace errors against an FP64 reference, an `ok` verdict and median time.

To trace the internal steps of an algorithm, start from the repository root:

```bash
python -m algorithm_laboratory.experiment --algorithm mine
```

Open [my_algorithm.py](my_algorithm.py), change a line, rerun the command, and
inspect the new folder under `laboratory_runs/`. Put `breakpoint()` in the loop
when you want to stop and examine live tensors. No web server is needed.

| File | What to do here |
|---|---|
| [my_svd.ipynb](my_svd.ipynb) | Paste your `my_svd(a, rank)` and get a metrics table, comparison and plots. |
| [scorecard.py](scorecard.py) | `evaluate` / `compare` any SVD function or builtin method on a standard suite; also a CLI. |
| [my_algorithm.py](my_algorithm.py) | Your working starter; replace the numerical steps. |
| [algorithms.py](algorithms.py) | Inspect/edit randomized SVD, power iterations, Jacobi sweeps and plane rotations. |
| [experiment.py](experiment.py) | Choose the matrix, capture intermediate state and run independent timings. |
| [parts.py](parts.py) | Test individual operations; the prepared example compares two QR implementations. |
| [specimens.py](specimens.py) | Prepare 11 small dense, sparse, clustered, complex, batched and custom-spectrum cases. |
| [study.py](study.py) | Sweep power iterations, oversampling and sketch seeds on identical input. |
| [laboratory.ipynb](laboratory.ipynb) | Interactive cells for running, inspecting arrays, checking errors and comparing steps. |
| [power.py](power.py) | Optional separate sustained-load measurement with NVIDIA telemetry. |
| [../svdbench/laboratory](../svdbench/laboratory) | Reusable matrix, trace, diagnostic, statistical and plotting tools. |

[The complete manual](../docs/ALGORITHM_LABORATORY.md) explains the measurement
contracts, saved files, GPU commands and ways to add your own instruments.

A [saved GPU example](../results/laboratory-example) is included so you can inspect
real intermediate arrays immediately. See [validation results](../results/laboratory-validation/README.md).
