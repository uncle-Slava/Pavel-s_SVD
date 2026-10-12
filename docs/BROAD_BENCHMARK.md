# Reproducing the larger-matrix benchmark

The suite in `configs/broad.json` defines 14 workloads and 143 method/matrix
comparisons. It includes full-spectrum decompositions and separately ranked top-k
approximations. Shapes extend to 2048×2048 square, 8192×512 tall full-spectrum,
8192×2048 rank-64, and 4096×2048 sparse inputs. Clustered spectra are tested in FP32
and FP64, with exact or nearly repeated singular values and conditions up to 1e10.

From the source checkout, using a CUDA-enabled Python environment:

```bash
python -m svdbench.broad --config configs/broad.json \
  --output-dir experiments/my-broad-run
python -m tools.supplement_broad experiments/my-broad-run \
  --requests configs/broad-supplement.json \
  --output experiments/my-broad-supplement
MPLCONFIGDIR=/tmp/svd-matplotlib python tools/analyze_broad.py \
  experiments/my-broad-run --supplement experiments/my-broad-supplement \
  --output results/my-broad-report
```

The [12 October results](BROAD_RESULTS.md) combine the main run with 26 follow-up
attempts, giving 169 attempts and 166 distinct settings. Three follow-ups retry
initial native-binding setup failures; the others add a dense-2048 control and
tune randomized iterations/oversampling. Every supplement reuses the exact cached
matrix and reference. It hard-links those files, so both experiment directories
must be on the same filesystem. The reporter uses the latest attempt for each
case/method ID without deleting earlier attempts. Use a new ID for a new parameter
setting. A fresh baseline with the binding present will not have the original
setup failures; its three repeated native calls are retained the same way.

Run from the checkout root, including `references/hetzner-cusolver_direct.py`,
to enable the native controls. The source installation and vendor library must
support the requested routines. The standalone package can run elsewhere, but
this larger suite intentionally targets NVIDIA CUDA and explicit cuSOLVER paths.

Output directories must be new. The runner stores every original matrix, complete
FP64 reference, subprocess log and result. An append-only `measurements.jsonl` file
records completed attempts immediately; `progress.json` identifies the current
attempt and `complete.json` confirms the whole suite finished. `environment.json`
records software, GPU, CPU, TF32 settings and a source fingerprint. Reference files
are cached only within that new experiment. The runner creates a fresh process for
each algorithm, releasing its GPU allocations before the next method.

Allow exclusive GPU access. The Hetzner run temporarily stops the idle playground
service and restarts it afterward; the Python runner itself does not manage services.
Ordinary benchmarking does not require root. Avoid editing source during a run.

The full 16-method standalone set is attempted on the 256×256 Gaussian case.
Larger full decompositions use a selected set of standalone families plus vendor
and library controls. Larger top-k cases focus on randomized, Krylov and Lanczos
methods. The cold-start standalone `subspace` and `randomized` variants currently
use the same numerical iteration; their measurements are not independent algorithms.

Each measurement has one warmup and three timed repetitions, followed by a separate
solve for accuracy validation. Median synchronized wall time is the primary speed
metric. Startup, input generation, CPU reference, transfers/preparation and validation
are excluded from solver timing. GPU allocator peaks are captured across timed calls.
Timed output allocations are included; prior output tensors are released between calls.

The configured 60-second worker limit includes all worker activity, so a timeout is
not equivalent to a 60-second single SVD. Reference preparation has a separate
600-second limit. Timeouts, numerical errors and failed accuracy gates remain in the
raw results and are excluded from the accepted speed/memory ranking.

Acceptance is deliberately stricter than reconstruction alone:

- Full FP32: relative reconstruction, spectrum error/sigma1, and normalized U/V
  orthogonality each ≤2e-4; full FP64: each ≤1e-9.
- Top-k: reconstruction ≤1.05 times the optimal FP64 reference tail, maximum relative
  error of the resolved leading k singular values ≤1%, plus the dtype's orthogonality
  limits. If the reference tail is zero, use the full reconstruction tolerance.
- All outputs: finite, nonnegative singular values in descending order.

The rank-k budget suits spectral approximation experiments; an application can choose
a different budget. The full-spectrum budget does not guarantee high relative accuracy
for every tiny singular value. That metric is always reported separately.

Memory comparisons use **peak allocated Torch GPU memory**, including resident input
and prepared buffers, rather than only the incremental workspace. It excludes CUDA
context/driver memory and allocations outside the Torch allocator. CPU reference
memory is excluded. Coarse 500 ms NVIDIA telemetry is saved separately and is not
used to rank per-call memory, energy or occupancy.

Sparse algorithms use COO products, but the harness retains the same dense GPU
input for every method and uses dense CPU references. This tests sparse arithmetic,
not the minimum memory footprint of an entirely sparse application. Sparse conversion
and sketch generation for the library controls are preparation work outside the timer.
The standalone sketch-generation cost remains inside its solver, as implemented.

The existing Xgesvdr wrapper allocates full square U/V storage even for truncated
outputs. Its peak describes the wrapper and must not be presented as an intrinsic
cuSOLVER memory requirement. Native wrappers include their workspace queries,
allocations and status synchronization inside each call. Explicit vendor top-k
controls perform a full solve and slice the result.

Randomized implementations do not share the same random sketch. The legacy native
wrapper does not expose its RNG seed. One input seed per case and three timing
samples provide a broad workload comparison, not a statistical characterization
over random inputs or an exhaustive performance/accuracy guarantee.
