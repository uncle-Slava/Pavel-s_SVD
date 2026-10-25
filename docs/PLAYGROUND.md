# SVD Lab: interactive matrix playground

Run all 16 standalone matrix algorithms, a sparse Lanczos adapter, and vendor
controls on identical inputs. Change a solver's Python implementation, run the
experiment again, and compare its time, reconstruction, spectrum, orthogonality,
and memory use. Each solve starts a fresh Python process, so solver edits take
effect on the next run without restarting the web server.

## Start on your laptop

From the repository, using Python 3.10 or newer:

```bash
python -m pip install -e .
python -m svdbench.playground.server
```

Open **http://127.0.0.1:8765**. The installed `svd-playground` command is equivalent.
The browser uses no external services or JavaScript dependencies. A CUDA/ROCm
PyTorch build exposes the GPU selector; CPU also works. Choose the appropriate
PyTorch build before installing this project.

If a GPU tunnel already occupies port 8765, start the laptop server with
`python -m svdbench.playground.server --port 8766` and open port 8766 instead.

## Run it on a remote GPU machine

Install the package on the GPU machine with a CUDA/ROCm PyTorch build and start the
server there. It listens on loopback only, so reach it through an SSH tunnel from your
laptop and leave the tunnel running:

```bash
# on the GPU machine
python -m svdbench.playground.server --port 8765

# on your laptop
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 <user>@<gpu-host>
```

Open **http://127.0.0.1:8765**. The header shows the GPU the server found. Use the
same port on both sides: the application validates the HTTP Host and Origin. If a
laptop playground already occupies 8765, stop that process first, or change the
server port and both tunnel ports together. Stop the tunnel with Ctrl-C.

To keep the server running, [deploy/svd-playground.service](../deploy/svd-playground.service)
is an example systemd unit. Edit its paths and user for your machine, then:

```bash
sudo cp deploy/svd-playground.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now svd-playground
journalctl -u svd-playground -n 30
```

`systemctl disable --now svd-playground` removes it again. The service never listens
on a public network interface. Ask the repository owner for access to the team's
shared GPU machine.

## Experiments to try

| Question | Matrix and settings |
| --- | --- |
| Does a proposal exploit low rank? | Low rank, 256×128, matrix rank 8, tail 0 or 1e-5, top-k 8; compare randomized, block Krylov, Jacobi, framework SVD. |
| Are nearly equal values a problem? | Clustered, float64, cluster size 4, relative spread 0 then 0.001; compare reconstruction and subspaces, not individual vector signs. |
| Does sparsity help? | Sparse, 128×64, density 0.01–0.2; compare sparse Lanczos, dense Lanczos, and framework SVD. Raise Lanczos steps when the spectrum decays slowly. |
| Where do Gram methods lose accuracy? | Geometric, reduced output, condition ratio 1e4–1e12, float32 versus float64; compare Gram + Jacobi and one-sided Jacobi. |
| How does size affect overhead? | Open Size sweep, enter `16, 32, 64, 128`, select the methods, and run. The latency chart uses each size as a separate experiment case. |
| Does batching change the comparison? | Set batch to 8 or 16 with small matrices. Metrics report the worst batch member; plots show the first. |
| What happens at rank zero? | Zero matrix with reduced or full output; inspect null-space completion and orthogonality. |

Generating rank and requested output rank are separate controls. A positive tail
means the low-rank family is approximately low rank. Cluster controls concern
**singular values**, rather than general matrix eigenvalues. Sparse input uses a
Bernoulli nonzero mask; the realized density can differ from the requested density.

Use **All** to select compatible methods, or start with the three-method **Core set**.
Explicit cuSOLVER drivers are available only on NVIDIA. Sparse Lanczos requires the
sparse family. Output choices are top-k, reduced SVD, full square bases, and values
only. Complex64/128 are supported. Iteration, oversampling, sweep limits, timeout,
thread count, warmup count, repetition count, scale and accuracy budget are editable.

Click a result row to inspect its spectrum, reconstruction/error heatmap, and full
diagnostics. History survives a server restart. Export JSON saves the configuration,
environment, metrics, source/input SHA-256 fingerprints and sampled plots; Matrix .npy downloads the actual input.
Import experiment accepts either an exported record or a raw configuration, such as
the files under [configs/playground](../configs/playground). Import configures a new
run; History opens the original measurements. Seeds alone do not guarantee identical
rounding across numerical libraries: preserve the .npy input and its SHA-256 for
cross-machine comparisons. The browser currently generates its inputs; arbitrary
.npy input is supported by the CLI corpus mechanism, not the upload control.

## How to interpret the results

- Solver timing excludes matrix generation, transfer/preparation, independent CPU
  reference, validation and plot construction. It includes Python dispatch and device
  synchronization. Both synchronized wall times and CUDA event times are exported.
  The table shows median wall time; these are Python implementations, not fused kernels.
- Warmups and repetitions are explicit. Every algorithm receives the same matrix
  bytes. The reference uses SciPy in float64/complex128 on the rounded input.
- Full/reduced SVD acceptance checks relative reconstruction, scaled orthogonality,
  sorted nonnegative values, and spectrum error normalized by sigma1. Values-only
  output checks the spectrum. Top-k acceptance uses at most 1.05 times the optimal
  reference tail plus the selected orthogonality budget. If the optimal tail is zero,
  relative reconstruction is used instead. This is not a guarantee of high relative
  accuracy for every tiny singular value; inspect the exported metrics.
- Peak +MiB is the Torch GPU allocator increment above prepared input. CPU peak
  memory and allocations outside Torch are not measured. The JSON also contains
  input/output bytes and allocator totals. Input storage describes the retained dense
  corpus, even for sparse matrices.
- Sparse Lanczos uses actual sparse COO matrix products. Conversion to sparse storage
  is preparation work outside the solver timer. Dense matrix/reference storage still
  exists, so this playground does not demonstrate end-to-end sparse memory savings.
- Vendor top-k controls compute a full decomposition and slice it. Standalone
  values-only modes may still form vectors internally. These differences are recorded.
- A completed experiment can contain timeouts, unsupported results or numerical
  failures. Rows outside the accuracy budget remain visible; speed comparisons should
  account for the required accuracy. A timeout covers each worker stage, including
  its Python startup, warmups, repetitions and validation.

## Limits and implementation

Interactive limits: each dimension ≤4096, min(m,n) ≤512, batch ≤64, total input
elements ≤4,194,304, full output bases ≤128 MiB, and at most eight sweep sizes.
Workers run sequentially; one experiment may be active at a time. Cancel terminates
the active child. A timed-out solver is killed and the next solver can proceed.
Interrupted state is preserved when the server restarts. For large benchmarks, use
the existing CLI and JSON configurations instead.

The server binds only to 127.0.0.1. Host/Origin checks and a per-process session token
protect compute requests from unrelated web pages. This is a single-user local tool,
not a multi-user authentication system. Do not expose it using a public reverse proxy.

Source entry points:

- `svdbench/playground/engine.py`: validation, solver adapters, plots, accuracy gates.
- `svdbench/playground/worker.py`: matrix/reference preparation and isolated solver jobs.
- `svdbench/playground/server.py`: HTTP API, experiment lifecycle, persistence, cancellation.
- `svdbench/playground/static/`: dependency-free browser interface.
- `svdbench/standalone/`: editable algorithms; see [STANDALONE.md](STANDALONE.md).
- `tests/test_playground.py`: generators, sparse modes, HTTP, export, lifecycle and cancellation.

For a variation of an existing algorithm, edit its implementation in
`svdbench/standalone/`, then repeat a saved configuration. To add a separately named
method, register it in `standalone/solvers.py` and add its label/group in
`playground/static/app.js`. Restart the service and reload the page after changing
the registry or the HTTP server. The engine derives its allowed methods from that
registry. Keep vendor calls in separately labeled controls so comparisons remain
clear. For arbitrary external plugin factories, use the CLI adapter contract in
[the standalone guide](STANDALONE.md).

When editing on the laptop while using a remote GPU machine for computation, first copy the
updated Python source to the server (run from the repository root):

```bash
rsync -az --exclude='__pycache__/' --exclude='._*' svdbench/ \
  <user>@<gpu-host>:<checkout>/svdbench/
```

Finish or cancel the current experiment before editing/syncing source. This keeps
each run's source fingerprint consistent with the code loaded by its workers.

Run `python -m unittest discover -s tests -v`. To exercise the standalone numerical
suite on CUDA/ROCm, prefix it with `SVD_TEST_DEVICE=cuda`. HTTP tests use an ephemeral
loopback port and bypass system web proxies.

## Validation on 12 October 2026

The complete suite passed **29 tests** on the RTX PRO 4000 server with `SVD_TEST_DEVICE=cuda`.
The seven new playground tests also passed on the macOS CPU environment.
A wheel build confirmed that HTML, CSS and JavaScript are included in the package.
Browser checks covered a desktop layout and a 390-pixel mobile layout (no horizontal
page overflow), GPU discovery, experiment submission, result selection, error
heatmaps, history and size-sweep case switching.

Six API experiments produced **40 GPU solver comparisons**, all passing their
configured accuracy gates. These are correctness and integration smoke tests,
not a broad performance conclusion or an exhaustive numerical validation.

| Experiment | Comparisons | Settings |
| --- | ---: | --- |
| Every available dense method | 20 | 20×12 FP64 Gaussian, reduced output; 16 standalone methods + four vendor controls |
| Sparse complex batch | 3 | 2×20×12 complex128, density 0.25, top-4, 12 Lanczos steps |
| Size sweep | 9 | 16, 32, 64 square, FP32 rank-4 + 1e-5 tail, three methods |
| Repeated singular values | 4 | 64×48 FP64, cluster size 6, zero spread, condition ratio 1e5 |
| Full wide batch | 2 | 2×12×20 complex64, full square vector bases |
| Values only | 2 | 32×20 FP64 geometric spectrum, condition ratio 1e5 |

Raw measurements: [playground-rtx4000-validation.json](../results/playground-rtx4000-validation.json).
The eight exact matrices and their hashes are in [corpus/playground](../corpus/playground).
Test logs are saved in `results/playground-*-tests*.log`.

To reproduce the API suite against a running playground:

```bash
python tools/check_playground.py --device cuda:0 --output results/my-playground-check.json
```

Choose a new output filename. On CPU use `--device cpu`; the explicitly NVIDIA
controls are omitted, so the comparison count differs. Run this suite while the
server is idle; it starts real experiments sequentially and preserves them in History.
