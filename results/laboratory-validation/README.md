# Algorithm laboratory validation — 12 October 2026

- [Complete Hetzner CUDA suite](hetzner-tests.log): **42 tests passed** in 12.431 s.
- [Final CUDA laboratory checks](hetzner-laboratory-final.log): **11 tests passed** after adding the stage-summary convenience method.
- [Local CPU laboratory tests](cpu-tests.log): **11 tests passed**.
- [Notebook smoke execution](notebook-smoke.log): every code cell executed on CPU.
- [GPU workflow smoke tests](hetzner-smoke.log): randomized SVD with power sampling,
  Jacobi with rotation snapshots, the QR parts bench and a four-trial parameter study.
- [Custom algorithm starter](custom-starter.log) and [specimen preparation](specimens.log)
  also ran successfully on CPU.

The tests cover trace/no-trace numerical equality, skipped diagnostic-only work,
immutable snapshot reload, snapshot budgets, failed-run retention, nonfinite
intermediates, early Jacobi stopping, complex/tall/wide/rank-deficient/zero inputs,
COO products, repeated-subspace boundaries, timing statistics and power integration.

The [committed GPU example](../laboratory-example) is a real run on a Gaussian
64×40 FP64 matrix, requesting rank 4 and three power iterations. It includes 29
events, raw tensors, input/reference, source snapshots, accuracy/timing results,
a plot and 200 ms NVIDIA power samples. Its large reconstruction residual mostly
reflects truncating a broad spectrum; inspect the optimal-tail ratio and individual
value errors rather than interpreting reconstruction alone as algorithm failure.

```python
from svdbench.laboratory import Run
run = Run("results/laboratory-example")
iterations = run.select(name="iterate")
q = run.tensor(iterations[1]["event_id"], "Q")
print(run.stage_statistics())
```

Larger local demonstrations are in `laboratory_runs/hetzner-jacobi` (375 events),
`hetzner-parts`, `hetzner-study`, and `specimens` (11 prepared cases). Hetzner's
copies remain under `/root/svd-standalone-20261011/laboratory_runs/` except the
specimen collection, which was generated locally. These working folders are
ignored by Git. The source snapshots in the committed example represent the
program at measurement time, before the later convenience stage-summary method
and specimen collection were added; they are intentionally preserved unchanged.

Power is a separate sustained, untraced load: 83 solver calls over 3.005 s; 15
in-window samples cover 2.802 s. Integrated gross device energy was 70.12 J over
that covered interval, averaging 25.02 W. This is a sampler smoke test, not a
competitive energy benchmark or per-iteration energy measurement.
