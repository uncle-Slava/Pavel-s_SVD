# Algorithm laboratory — editable Python experiments

The repository now has three complementary ways to work: the web playground for
quick comparisons, the algorithm laboratory for inspecting/changing Python code,
and benchmark/power tools for measuring execution. The laboratory has no web
dependency. Its scripts are deliberately small editable programs, backed by
reusable instruments in `svdbench.laboratory`.

## Checking an SVD: use svdlab

To score a whole SVD, or one replaced step, as a black box (metrics, verdict, timing, GPU
check, C++/CUDA files), use `python -m svdlab check`; see the
[developer guide](DEVELOPER_GUIDE.md). `algorithm_laboratory.scorecard` is the old import
path of `svdlab.scorecard` and still works. The rest of this manual is about tracing an
algorithm's internal steps.

## Start with your own algorithm

Use the project's Python environment (Python 3.10+ for installation). From the
repository root:

```bash
python -m pip install -e .
python -m algorithm_laboratory.experiment --algorithm mine
```

Edit [algorithm_laboratory/my_algorithm.py](../algorithm_laboratory/my_algorithm.py).
It contains a working sketch → orthogonalization → projected SVD → lifting
pipeline, with no power iterations. It is a starting point to modify, not a claim
of optimal accuracy. Add loops, change formulas, replace primitives, or insert
`breakpoint()` and use Python's debugger. Ordinary IDE debugging works as well.

The installed `svd-laboratory` command is equivalent to the module command.
Output directories are created exclusively; choose a new path for each run.
Without `--output`, a timestamped directory is created under `laboratory_runs/`.
These working results are ignored by Git so large snapshot collections are not
accidentally committed.

For optional notebook/DataFrame tools:

```bash
python -m pip install -e '.[laboratory]'
python -m jupyterlab algorithm_laboratory/laboratory.ipynb
```

Use the appropriate existing CUDA/ROCm Torch build; do not replace it with a
different accelerator build just to install the laboratory.

## Prepared experiments

```bash
# Prepare 11 named specimens and independent references, then reuse their bytes.
python -m algorithm_laboratory.specimens --output laboratory_runs/specimens

# Trace each power iteration, with standalone QR and projected SVD by default.
python -m algorithm_laboratory.experiment --algorithm randomized \
  --m 128 --n 64 --family low_rank --matrix-rank 8 --rank 8 --iterations 4

# Inspect sweeps and each group of independent Jacobi rotations.
python -m algorithm_laboratory.experiment --algorithm jacobi \
  --family clustered --m 24 --n 16 --condition 1e8 --record-rotations

# Stop after exactly one sweep even if unconverged; inspect the resulting errors.
python -m algorithm_laboratory.experiment --algorithm jacobi \
  --family gaussian --m 24 --n 16 --max-sweeps 1 --allow-unconverged

# Change one operation at a time; compare QR on the same ill-conditioned input.
python -m algorithm_laboratory.parts --output laboratory_runs/qr-parts

# Separate sketch-seed variation from repetitions of a timing sample.
python -m algorithm_laboratory.study --iterations 0,1,2,4 --oversamples 4,8 \
  --seeds 3,7,11 --output laboratory_runs/power-iteration-study

# Use the exact bytes of a matrix from the broad benchmark.
python -m algorithm_laboratory.experiment --input \
  experiments/broad-20261012/lowrank2048x1024_k32/matrix.npy \
  --rank 32 --device cuda:0 --primitives library --snapshots stats

# Actual COO products on a sparse input (reference generation remains dense).
python -m algorithm_laboratory.experiment --family sparse --density .01 \
  --sparse-products --primitives library --iterations 8
```

`--rank` is the requested output rank for randomized/custom algorithms;
`--matrix-rank` controls the generating low-rank matrix. Jacobi returns all
singular triplets. `--input` loads a `.npy` matrix and overrides generating shape,
dtype, family and spectrum settings. All original matrix families are available:
Gaussian, low-rank with a configurable tail, clustered, geometric, sparse and zero.
Real/complex FP32/FP64 are supported. The editable starter solvers use one matrix
at a time; the matrix/reference tools also support batches, which you can loop over.

## Add measurements anywhere in your code

```python
from svdbench.laboratory import MatrixCase, Trace, basis_quality

case = MatrixCase.create(m=128, n=64, family="low_rank", rank=8, seed=23)
a = case.tensor("cpu")

with Trace("laboratory_runs/my-change", case, parameters={"idea": "new QR"},
           source_files=["algorithm_laboratory/my_algorithm.py"]) as trace:
    with trace.stage("matrix_product", iteration=0):
        y = a @ a.mH
    trace.checkpoint("after_product", iteration=0, tensors={"Y": y},
                     scalars={"my_measurement": y.abs().max()})
```

Use a stage for one non-overlapping computational block. Put checkpoints after
the stage. Nested measured stages and checkpoints inside stages are rejected
because they would contaminate timing/memory accounting.

At each checkpoint you can record:

- Arbitrary scalar measurements and named tensors; immutable CPU snapshots.
- Shape, dtype, bytes, zeros/nonfinite counts, norm, magnitude extrema, mean,
  standard deviation and quantiles; real and imaginary means for complex data.
- `factors=(U, s, Vh)` for independent reconstruction, U/V orthogonality, left/right
  triplet residuals, sorted/nonnegative values, spectrum errors and optimal-tail ratio.
- Full vectors of reference singular values, absolute/relative value errors and
  a mask identifying values resolved by the FP64 reference.
- `basis_quality(case, Q)` for basis orthogonality, projection residual and the
  largest principal angle to the reference leading subspace when uniquely defined.
- `residual_arrays(case, factors)` for elementwise reconstruction/triplet residuals,
  per-triplet norms and U/V Gram errors. Pass the returned dictionary as `tensors`
  to save every entry. These diagnostic arrays can be large.

`trace.record("my_metric", iteration=i, change=delta, active=count)` is useful for
cheap scalar observations. Guard diagnostic-only computation with
`if trace.wants(i): ...`; the uninstrumented `NullTrace` always returns false.
The randomized example uses this to skip diagnostic projected SVDs during timing.

Algorithms use `orthogonalize()` and `small_svd()` in the editable file. You can
replace either independently, or supply `orthogonalizer=my_qr` and
`core_solver=my_svd` to `randomized`. `primitives="library"` selects Torch controls;
the default uses the existing standalone arithmetic implementations.

The other 16 standalone variants remain available through
`svdbench.standalone.svd`. The laboratory does not automatically capture every
local variable inside them: add stages/checkpoints around the specific internal
parts you want to study. Jacobi and randomized SVD provide complete worked
examples of explicit internal instrumentation.

## Inspect, compare and reuse a run

The repository includes [a complete saved GPU run](../results/laboratory-example).
Load `Run("results/laboratory-example")` to inspect its arrays immediately without
running an algorithm. Additional downloaded demonstrations and a prepared specimen
collection are in the local `laboratory_runs/` directory.

```python
from svdbench.laboratory import Run, MatrixCase, check_accuracy

run = Run("laboratory_runs/my-run")
steps = run.select(kind="checkpoint", name="iterate")
q0 = run.tensor(steps[0]["event_id"], "Q")
q1 = run.tensor(steps[1]["event_id"], "Q")
print(steps[1]["accuracy"]["worst"])

# Projectors avoid sign/phase ambiguity when comparing bases.
import numpy as np
print(np.linalg.norm(q1 @ q1.conj().T - q0 @ q0.conj().T))

print(check_accuracy(steps[-1]["accuracy"],
                     residual_over_optimal_tail=1.05,
                     resolved_spectrum_max_relative_error=.01))
run.plot()                       # Regenerate trace.png from saved JSON.
print(run.stage_statistics())     # Counts, totals, raw samples, median/MAD per operation.
df = run.dataframe("checkpoint") # Optional pandas; all nested fields are flattened.
```

When the reference tail is zero, `residual_over_optimal_tail` is unavailable;
choose a reconstruction tolerance instead. `check_accuracy` fails unavailable
requested metrics, so missing measurements do not silently pass. Intermediate
iterates are often intentionally inaccurate; a completed trace does not mean
the final result met a numerical budget.

Every experiment directory contains `input.npy`, `reference.npz`, `run.json`,
append-only `events.jsonl`, `snapshots/*.npy`, saved `source/*.py`, `result.json`
and `trace.png`. `run.json` records dtype/parameters, input hash, package/device
details and source hashes. `result.json` holds final accuracy and every raw timing
sample. Failed traces retain their completed events and snapshots; a divergent
factor checkpoint records `validation_error` and preserves its arrays.

Snapshots default to a **64 MiB array payload budget**. Statistics continue when
the budget is exhausted, and each tensor records why its array was not saved.
`--snapshots stats` disables intermediate array files; `--trace-every 2` samples
checkpoints every other iteration. Stage measurements are still recorded. The
budget excludes original input/reference files, source and JSON, and is not a
peak-RAM limit. Statistics/validation still copy full tensors to the CPU. Use
small matrices while exploring internals; exact FP64 references are dense and
expensive. The API also bounds event counts and records any dropped events.

## Timing, statistics, memory and power

An experiment first performs an inspection solve, then releases its GPU outputs
and benchmarks the same code with `NullTrace`. Benchmark time excludes input
generation, transfer, CPU reference, snapshots, plotting and accuracy checks.
It includes work the algorithm performs itself, including sketch generation and
convergence checks. One process retains warm library caches. The inspection pass
changes synchronization/cache behavior; its stage timings are diagnostic, not a
substitute for the independent uninstrumented run.

The statistical toolbox preserves samples and reports mean, median, sample
standard deviation, MAD, extrema, p10/p90 and a reproducible bootstrap interval
for the median when at least five samples exist. Timing samples can be correlated;
the interval is not a guarantee over different matrices or machines. A study
separates repeated timings from multiple sketch seeds and preserves failed trials.
It holds input bytes fixed across those trials; changing matrix seeds requires
an outer loop or additional study runs.

CUDA/ROCm stages report synchronized wall and stream-event time, Torch allocated
peaks/increments and benchmark reserved memory. CPU memory, driver/context and
non-Torch allocations are unmeasured, represented by nulls. Sparse modes retain a
dense CPU input/reference but pass a COO GPU matrix to the solver. No pure-sparse
end-to-end memory claim follows from this reference-based experiment.

For a **separate sustained NVIDIA load** after the timing pass:

```bash
python -m algorithm_laboratory.experiment --device cuda:0 --primitives library \
  --m 512 --n 256 --rank 16 --matrix-rank 16 --power-seconds 5
```

`power.jsonl` records physical GPU UUIDs, approximate host receipt times, power,
utilization and device memory at 200 ms intervals. The report integrates power
over covered in-window samples only; it does not extrapolate missing boundaries.
Energy is gross device energy with no idle subtraction. Sensor averaging, other
processes and sampling overhead matter. It is **not** per-iteration energy. The
existing `tools/run_monitored.py` remains available for the main benchmark CLI.

## Remote GPU machine

On a GPU machine, install the package in an environment with a CUDA/ROCm PyTorch build
and run the same command with a GPU device:

```bash
python -m algorithm_laboratory.experiment \
  --device cuda:0 --algorithm randomized --output laboratory_runs/my-gpu-change
```

To edit on a laptop and run on the GPU machine, sync the changed Python files first; see
the [local-to-GPU workflow](PLAYGROUND.md). This laboratory has been tested on CPU and
NVIDIA CUDA. ROCm uses the portable Torch paths but remains untested on AMD hardware.
Ascend timing/tracing and power instrumentation are not implemented here.
