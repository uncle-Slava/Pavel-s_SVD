# Algorithm laboratory

A Python workbench for changing numerical algorithms and inspecting their steps.
It runs independently of the web playground.

**To write an SVD and check it, start with `svdlab`, not here:** `python -m svdlab check
examples/my_svd.py`. See the [quickstart](../README.md#quickstart) and the
[developer guide](../docs/DEVELOPER_GUIDE.md). This folder is for the next step: looking
inside an algorithm while it runs.

To trace the internal steps of an algorithm, start from the repository root:

```bash
python -m algorithm_laboratory.experiment --algorithm mine
```

Open [my_algorithm.py](my_algorithm.py), change a line, rerun the command, and
inspect the new folder under `laboratory_runs/`. Put `breakpoint()` in the loop
when you want to stop and examine live tensors. No web server is needed.

| File | What to do here |
|---|---|
| [my_svd.ipynb](my_svd.ipynb) | Notebook version of `svdlab check`: paste `my_svd(a, rank)`, get tables and plots. |
| [scorecard.py](scorecard.py) | Old import path of `svdlab.scorecard`, kept so existing code keeps working. |
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
