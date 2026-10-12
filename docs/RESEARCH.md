# Research assessment — 11 October 2026

The strongest research target is an **accuracy-constrained choice of SVD algorithm for a
specific workload**. There is no single decomposition implementation that should win on
large dense matrices, tiny independent batches, ill-conditioned spectra, and low-rank data.
Start with three tracks: full spectrum; approximate top-k; batched small matrices. Add tensor
compression as an application track, with its own reconstruction and memory objectives.

This assessment combines current primary documentation, inspection of both supplied servers,
review of the user's existing Hetzner project, and new measurements. It does not claim an
exhaustive literature review or results on unavailable AMD/Ascend hardware.

## Clarifying the premise

For a dense m-by-n matrix, with r=min(m,n), conventional reduced SVD takes order m*n*r
work: cubic for square matrices. Full rectangular singular-vector bases can cost substantially
more to form/store. SVD is versatile, but is neither the universally hardest linear-algebra
operation nor always the appropriate one: QR, Cholesky, iterative solves, and partial
decompositions solve narrower problems more cheaply. LAPACK distinguishes QR-iteration
and divide-and-conquer SVD drivers. [LAPACK overview](https://www.netlib.org/lapack/lug/node32.html)

Iteration itself is not the GPU obstacle. Dependencies between small operations, global
synchronization, reductions, panel factorization, and movement between processors are the
obstacles. Iterations dominated by large matrix products can work well. A recent research
direction keeps panel and bidiagonal divide-and-conquer work on the GPU; this deserves a
baseline before presenting a new full-SVD design as novel. The paper's reported speedups
are not predictions for these servers. [Liu et al., 2025](https://arxiv.org/abs/2508.11467)

For dense randomized rank-k SVD, let ell=k+p. A basic range finder uses order m*n*ell work
per pass, plus QR and the small solve. Power iterations add passes and must include
reorthogonalization. Low rank, spectral decay, oversampling, and target tolerance determine
whether this is attractive. A small residual is not expected when k discards most of a
flat spectrum. [Halko, Martinsson and Tropp](https://arxiv.org/abs/0909.4061)

## Hardware actually available

| Host | Verified hardware/software | Role |
|---|---|---|
| Yadro | RTX PRO 6000 Blackwell Server Edition; nvidia-smi 97,887 MiB; compute capability 12.0; 4 AMD EPYC Genoa virtual CPUs; about 58 GiB RAM; Debian; system CUDA toolkit 13.3 | Large-memory NVIDIA work; current pilot shared the GPU with another process |
| Hetzner | RTX PRO 4000 Blackwell SFF Edition; 24,467 MiB; compute capability 12.0; 70 W power limit; i5-13500, 20 logical CPUs; about 62 GiB RAM; Ubuntu | NVIDIA kernel development and smaller benchmarks; idle GPU before the run |
| AMD | No AMD GPU found in either server | Requires a separate supported ROCm GPU host |
| Orange Pi | User reports Ascend310B; currently powered off | Exact SoC, RAM, firmware and CANN remain unverified |

The CPU vendor on Yadro does not make it an AMD GPU platform. Driver/toolkit versions also
do not identify the library used by PyTorch: both new native runs loaded cuSOLVER **12.0.4**
from the PyTorch CUDA-13 wheel stack. The system toolkit is a separate installation.
The two PyTorch builds were 2.14.0+cu130 and 2.14.1+cu130, respectively. All measured
environment details are embedded in the JSONL records.

## Vendor algorithm map

| NVIDIA entry point | Documented family | Main comparison |
|---|---|---|
| gesvd / Xgesvd | QR-based SVD | Full spectrum and difficult spectra |
| gesvdj / gesvdjBatched | Jacobi | Sweep/tolerance tradeoffs; tiny batches |
| gesvdaStridedBatched | Gram-matrix approximation | Tall matrices; small-value accuracy |
| Xgesvdp | Polar decomposition plus eigensolve | Full spectrum; record perturbation |
| Xgesvdr | Randomized SVD | Small requested rank; vary oversampling/iterations |

The batched Jacobi API restricts **both** dimensions to at most 32. Workspace queries and
convergence information belong in every native experiment. These are documented algorithm
families, not a reconstruction of NVIDIA's private kernels or scheduling heuristics.
[NVIDIA cuSOLVER documentation](https://docs.nvidia.com/cuda/cusolver/index.html)

PyTorch exposes gesvd, gesvdj and gesvda; its default may fall back from Jacobi to QR.
Full, reduced, and singular-values-only output are different experiments. Explicit driver
selection is a cuSOLVER feature, not a cross-vendor algorithm selector.
[PyTorch SVD API](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd)

AMD's native baselines should include rocSOLVER gesvd, gesvdj, gesvdx, and the applicable
batched/strided variants. Subset extraction with gesvdx does not by itself promise randomized
rank-k complexity. rocSOLVER's source is available in `ROCm/rocm-libraries`; use the exact
installed release's source and tuning constants. This permits genuine implementation-level
inspection beyond what a closed vendor library exposes.
[AMD API summary](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/intro.html),
[source/build instructions](https://rocmdocs.amd.com/projects/rocSOLVER/en/latest/installation/build.html)

rocSOLVER workspaces are managed through the rocBLAS handle. Measure its memory-query
result and compare automatic allocation with a preallocated user-owned workspace; automatic
growth can synchronize execution. The current Torch-only ROCm path does not implement these
native queries. [AMD memory model](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/howto/memory.html)

Ascend has an aclnnSvd API, but the published CANN 9.0.0 operator support table marks Atlas
200I/500 A2 inference products unsupported, while supporting A2/A3 training/inference
families. The 310B ops installation is associated with the former family. Thus support on
larger Ascend NPUs must not be assumed on the Orange Pi. Recheck the installed release on
the board. [CANN SVD support table](https://gitcode.com/cann/ops-math/tree/9.0.0/math/svd),
[310B package mapping](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/softwareinst/instg/instg_0008.html)

## Algorithm families worth comparing

| Family | Attractive workload | Main risk / resource constraint |
|---|---|---|
| Bidiagonal reduction + QR | Reliable full-spectrum baseline | Panels, synchronization, back-transformation |
| Bidiagonal divide-and-conquer | Larger full SVD | Workspace, secular solves, efficient GPU reduction |
| One-sided/block Jacobi | Accuracy-sensitive problems; parallel small blocks | Sweep count, communication and stopping criteria |
| QR/TSQR then SVD of R | Tall-skinny matrices | QR cost; distributed/streaming reductions |
| Gram + Hermitian eigensolve | Well-conditioned tall matrices; dominant subspace | Squares condition number; null-space recovery |
| Polar + eigensolve | GEMM/QR-heavy full-spectrum research | Scaling, convergence, perturbation and tiny values |
| Randomized/subspace/block Krylov | k much smaller than r | Spectrum-dependent approximation; extra passes |
| Lanczos/Golub–Kahan bidiagonalization | Sparse/matrix-free partial SVD | Reorthogonalization and irregular memory access |
| Updating/warm-start methods | Slowly changing matrices | Drift, loss of orthogonality, restart policy |

The Gram route changes conditioning: kappa(A* A)=kappa(A)^2. Its dominant subspace may still
be useful after its smallest singular values have become unreliable. Do not equate success
in a beamforming metric with a valid full SVD. A scalar/column-wise Python Jacobi prototype
also does not measure the potential of a fused block kernel.

For polar methods, a good orthogonal factor alone is not the complete SVD: the remaining
Hermitian spectral problem still has a cost. Mixed precision needs a residual check in higher
precision and an explicit correction/fallback policy. GEMM-heavy or matmul-only structure
is a promising implementation property, not evidence of novelty or correctness by itself.

## Memory and resource requirements

For one real matrix with b bytes per scalar, storage lower bounds excluding workspace are:

- Input: b*m*n.
- Reduced full-spectrum outputs: b*(m*r + n*r + r).
- Full outputs: b*(m*m + n*n + r).
- Rank-k outputs: b*(m*k + n*k + k).

Complex U/V/input take complex-scalar storage while the singular values remain real.
Batching multiplies storage. Input restoration, layout copies, reference arrays, internal
promotions and workspaces can dominate these lower bounds.

Example derived from these formulas: a 131072-by-1024 FP32 input occupies 512 MiB.
Full U alone occupies **64 GiB**; reduced U occupies 512 MiB; rank-64 U occupies 32 MiB.
This output choice can decide feasibility before any kernel optimization. The existing
Hetzner randomized wrapper allocates m-by-m U even for top-k, so its 262144-row complex64
case requests 512 GiB for that array alone. This establishes a wrapper allocation issue,
not a measured requirement of the mathematical algorithm or vendor workspace.

Report five resource quantities separately: device input/output bytes; queried native
workspace; allocator/whole-process peak; host memory; transfers. Add kernel launch count,
achieved bandwidth, occupancy, tensor/vector utilization, host synchronization, and energy
only when measured with an appropriate profiler. The pilot provides allocator memory and
coarse whole-device telemetry; it does not provide those deeper counters.

## Python-to-native design

Keep the benchmark controller, corpus, accuracy checks and result schema in Python.
Implement each candidate as a small computation graph over GEMM, QR, reductions,
orthogonalization, rotations and a small spectral solve. Replace one primitive at a time:

| Layer | NVIDIA | AMD | Ascend310B experiment |
|---|---|---|---|
| Python orchestration | PyTorch CUDA | PyTorch ROCm | PyACL/CANN, or verified TorchNPU support |
| Vendor baseline | cuSOLVER | rocSOLVER / hipSOLVER | CPU baseline; aclnnSvd only if actually supported |
| Matrix products | cuBLAS/cuBLASLt | rocBLAS/hipBLASLt | Supported CANN matmul/Cube operations |
| Custom kernels | CUDA C++, optionally Triton | HIP C++, supported Triton path | Ascend C with explicit tiling/buffering |
| Trace | Nsight Systems/Compute | ROCm profiler tools | msprof and operator placement |

This is a shared specification and test harness, not automatic language conversion.
Warp/wave scheduling, local memory, synchronization and NPU pipelines require different
implementations. Start Ascend with GEMM/reduction/transpose/dtype capability tests, then a
hybrid rank-k algorithm: supported NPU products plus a small CPU QR/SVD. Measure every
transfer and label it **hybrid**. Move the small solve onto the NPU only if it improves
end-to-end time at the same accuracy. INT8 TOPS is not a prediction of FP32/FP64 SVD speed.

## Tensor track

A tensor can represent a batch of independent matrices or a higher-order object whose
modes must be factorized. These are different tasks. For Tucker/HOSVD/ST-HOSVD, record each
unfolding's shape, strides, materialization cost and intermediate sizes, then measure the
final tensor reconstruction error. Later compare tensor-train SVD if that format matches
the application. A reshape is sometimes a view; a transpose followed by a contiguous
unfolding may require a full copy. Randomized Tucker algorithms can preserve useful
structure and reduce data passes. [Minster, Saibaba and Kilmer](https://arxiv.org/abs/1905.07311)

The new harness currently handles matrices and one batch dimension. It does not yet execute
an end-to-end Tucker or tensor-train decomposition. The existing Hetzner project has tensor
unfolding generators worth reusing after checking their memory behavior.

## Recommended next experiments

1. **Establish reproducibility.** Use the same corpus hashes and, where possible, matching
   framework/library versions. Repeat after isolating Yadro from other GPU activity and
   accounting for its four-vCPU VM. Randomize algorithm order and repeat independent runs.
   Preserve normal clocks/power limits unless the experiment explicitly changes them.
2. **Choose accuracy budgets before winners.** Example research grid: reconstruction and
   orthogonality targets 1e-3, 1e-5, 1e-7; separate small-singular-value relative accuracy.
   For top-k, compare residual/optimal-tail and task loss; for repeated values compare the
   entire invariant subspace. These are proposed budgets, not achieved guarantees.
3. **Expand sizes and distributions.** Square 32–16384 subject to memory preflight; aspect
   ratios 1, 4, 16, 256; batches 1–4096; FP32/FP64/complex64/complex128; flat, decaying,
   clustered, scaled and exactly rank-deficient inputs. Add real model weights/activations.
   Sweep k/r and at least several independent sketch seeds, not just repetition of one sketch.
4. **Profile and improve targeted kernels.** Prioritize fused small-batch kernels and
   rank-adaptive randomized methods. For full-spectrum work, compare block Jacobi, polar,
   TSQR and GPU-centered bidiagonal divide-and-conquer. Keep cold-start, ordinary-call and
   graph-replay results separate. Exclude numerically invalid results from speed rankings.
5. **Add AMD hardware and native adapter.** First characterize the available supported GPU;
   CDNA/RDNA and FP64 throughput matter. Record actual rocSOLVER entry points and query
   workspace. Reuse identical acceptance tests before optimizing HIP kernels.
6. **Bring up the Orange Pi.** Once powered on, verify its exact SoC/RAM/CANN versions and
   primitive support, run CPU controls, then the explicitly hybrid rank-k path. Quantify
   error and energy per decomposition within a board-sized memory budget.

The most promising proposal to test first is **an adaptive randomized/block-subspace SVD
with stable reorthogonalization**, followed by **a fused tiny-matrix kernel with a clear
accuracy envelope**. A universal replacement for every full SVD is a much larger claim.
