# Larger-matrix SVD benchmark — Hetzner, 12 October 2026

166 distinct method/matrix comparisons on 14 workload cases, from 169 total attempts. Hardware: NVIDIA RTX PRO 4000 Blackwell SFF Edition; Torch 2.14.1+cu130; CUDA build 13.0.

Execution statuses: {'timeout': 4, 'ok': 161, 'error': 1}. Accuracy gates passed: 91.

## Fastest and lowest-memory accepted results

Winners are the lowest observed medians or allocator peaks among methods that pass the declared accuracy gates. Full and top-k outputs are ranked separately by workload. Three timed calls follow one warmup; small timing differences are not statistically established.

| Workload | Fastest accepted (ms) | Lowest peak allocated GPU memory (MiB) | Accepted / attempted |
|---|---|---|---|
| dense256 | cuSOLVER Xgesvdp **4.668** | cuSOLVER Jacobi **1.57** | 17/22 |
| dense512 | cuSOLVER Xgesvdp **10.816** | cuSOLVER QR **8.76** | 5/10 |
| dense1024 | cuSOLVER Xgesvdp **26.830** | cuSOLVER QR **33.27** | 4/10 |
| dense2048 | cuSOLVER gesvda **228.791** | cuSOLVER QR **130.30** | 2/7 |
| tall8192x512 | Library Gram **5.167** | cuSOLVER Xgesvdp **77.26** | 8/10 |
| cluster512_fp32 | cuSOLVER Xgesvdp **10.813** | cuSOLVER QR **8.76** | 6/10 |
| cluster512_fp64 | cuSOLVER Xgesvdp **54.938** | cuSOLVER Jacobi **12.27** | 7/10 |
| cluster1024_fp64 | cuSOLVER Xgesvdp **184.375** | cuSOLVER Jacobi **48.56** | 4/6 |
| lowrank2048x1024_k32 | Library randomized q2 **2.178** | Library randomized q2 **44.51** | 7/11 |
| lowrank8192x2048_k64 | Library randomized q2 **6.509** | Library randomized q2 **107.78** | 6/10 |
| dense1024_top32 | Library randomized q8 **9.355** | cuSOLVER Xgesvdr (legacy wrapper) q8 **24.91** | 5/12 |
| sparse2048x1024_d01 | Library Gram **10.021** | cuSOLVER Xgesvdr (legacy wrapper) q8 **45.66** | 8/16 |
| sparse4096x2048_d001 | Sparse library randomized q8 **11.781** | Library randomized q8 **73.26** | 7/16 |
| sparse4096x2048_d01 | Library randomized q8 **18.512** | Library randomized q8 **75.63** | 5/16 |

## Accuracy observations

| Workload | Lowest relative reconstruction error among accepted | Smallest maximum resolved singular-value relative error among accepted |
|---|---|---|
| dense256 | cuSOLVER gesvda: 4.342e-08 | cuSOLVER gesvda: 5.254e-08 |
| dense512 | cuSOLVER gesvda: 4.295e-08 | cuSOLVER gesvda: 5.510e-08 |
| dense1024 | cuSOLVER gesvda: 4.327e-08 | cuSOLVER gesvda: 5.831e-08 |
| dense2048 | cuSOLVER gesvda: 4.266e-08 | cuSOLVER gesvda: 5.568e-08 |
| tall8192x512 | cuSOLVER gesvda: 4.380e-08 | cuSOLVER gesvda: 5.870e-08 |
| cluster512_fp32 | cuSOLVER gesvda: 4.482e-08 | cuSOLVER gesvda: 5.824e-08 |
| cluster512_fp64 | cuSOLVER QR: 4.215e-15 | Standalone bidiagonal D&C: 1.072e-07 |
| cluster1024_fp64 | cuSOLVER Xgesvdp: 6.918e-15 | cuSOLVER QR: 1.235e-11 |
| lowrank2048x1024_k32 | Library Gram: 1.175e-03 | Library Gram: 1.152e-06 |
| lowrank8192x2048_k64 | Library Gram: 1.188e-03 | Library Gram: 3.579e-07 |
| dense1024_top32 | Library Gram: 9.424e-01 | Library Gram: 7.106e-08 |
| sparse2048x1024_d01 | Library Gram: 9.500e-01 | Library Gram: 1.418e-07 |
| sparse4096x2048_d001 | Library Gram: 9.597e-01 | Library Gram: 1.293e-07 |
| sparse4096x2048_d01 | Library Gram: 9.537e-01 | Library Gram: 2.818e-07 |

These two accuracy metrics can select different methods. Top-k reconstruction includes the irreducible discarded tail; compare its ratio to the optimal reference tail. Tiny differences near the reference or precision floor should not be treated as meaningful accuracy wins.

## Standalone implementations

| Workload | Fastest accepted standalone | Wall ms | Times slower than fastest accepted overall |
|---|---|---:|---:|
| dense256 | polar_polynomial | 691.662 | 148.2× |
| dense512 | Standalone bidiagonal D&C | 1603.870 | 148.3× |
| dense1024 | Standalone bidiagonal D&C | 3256.312 | 121.4× |
| dense2048 | No accepted standalone result | — | — |
| tall8192x512 | Standalone Gram/Jacobi | 1596.899 | 309.1× |
| cluster512_fp32 | Standalone bidiagonal D&C | 1516.436 | 140.2× |
| cluster512_fp64 | Standalone Newton–Schulz | 3676.156 | 66.9× |
| cluster1024_fp64 | Standalone Jacobi | 11142.913 | 60.4× |
| lowrank2048x1024_k32 | Standalone subspace | 130.846 | 60.1× |
| lowrank8192x2048_k64 | Standalone randomized | 245.199 | 37.7× |
| dense1024_top32 | Standalone dense Lanczos | 443.355 | 47.4× |
| sparse2048x1024_d01 | Standalone dense Lanczos | 446.552 | 44.6× |
| sparse4096x2048_d001 | Standalone dense Lanczos | 438.397 | 37.2× |
| sparse4096x2048_d01 | No accepted standalone result | — | — |

## Sensitivity to the FP32 accuracy budget

This secondary view relaxes only the full-spectrum FP32 reconstruction, spectrum/sigma1 and orthogonality limits from 2e-4 to 1e-3. It does not alter the primary ranking or the raw outcomes.

| Full FP32 workload | Fastest under 2e-4 | Fastest under 1e-3 |
|---|---|---|
| dense256 | cuSOLVER Xgesvdp 4.668 ms | cuSOLVER Xgesvdp 4.668 ms |
| dense512 | cuSOLVER Xgesvdp 10.816 ms | cuSOLVER Xgesvdp 10.816 ms |
| dense1024 | cuSOLVER Xgesvdp 26.830 ms | cuSOLVER Xgesvdp 26.830 ms |
| dense2048 | cuSOLVER gesvda 228.791 ms | cuSOLVER gesvda 228.791 ms |
| tall8192x512 | Library Gram 5.167 ms | Library Gram 5.167 ms |
| cluster512_fp32 | cuSOLVER Xgesvdp 10.813 ms | cuSOLVER Xgesvdp 10.813 ms |

## Failed accuracy gates and incomplete measurements

Timeouts are **whole-worker limits**, including startup, warmup, three timed solves, an additional validation solve and CPU validation. They are not per-SVD lower bounds. Failed/timeout rows do not enter speed or memory rankings.

- `dense256` / `bidiag_qr`: timeout; Worker timeout; limit 60 seconds covers startup, warmup, timed repetitions and validation.
- `dense256` / `block_jacobi`: timeout; Worker timeout; limit 60 seconds covers startup, warmup, timed repetitions and validation.
- `dense256` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.0010948336184815072 > 0.0002; reconstruction_relative_fro=0.00039156724752984444 > 0.0002; spectrum_error_over_sigma1=0.0030206901598446474 > 0.0002
- `dense256` / `gram_dc`: ok; reconstruction_relative_fro=0.00038781732350075325 > 0.0002; spectrum_error_over_sigma1=0.0030206901598446474 > 0.0002
- `dense256` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.0019110614346791127 > 0.0002
- `dense512` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.0013160601369907544 > 0.0002; reconstruction_relative_fro=0.0010024308183207726 > 0.0002; spectrum_error_over_sigma1=0.007745253454271697 > 0.0002
- `dense512` / `polar_newton_schulz`: ok; reconstruction_relative_fro=0.00022323513130592673 > 0.0002
- `dense512` / `vendor_gesvdj`: ok; orthogonality_v_fro_over_sqrt_rows=0.00024538977454565054 > 0.0002
- `dense512` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.026885604332891045 > 0.0002
- `dense512` / `qr_library`: ok; orthogonality_v_fro_over_sqrt_rows=0.00022038182391743072 > 0.0002
- `dense1024` / `jacobi`: ok; orthogonality_v_fro_over_sqrt_rows=0.00025987303829000006 > 0.0002; reconstruction_relative_fro=0.00023124353615064884 > 0.0002
- `dense1024` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.0015557446384733692 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0002596665802263636 > 0.0002; reconstruction_relative_fro=0.0014192566174850985 > 0.0002; spectrum_error_over_sigma1=0.01033413022258993 > 0.0002
- `dense1024` / `polar_newton_schulz`: ok; orthogonality_u_fro_over_sqrt_cols=0.00025226516189009085 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.00025226779226374655 > 0.0002; reconstruction_relative_fro=0.00047731576334753673 > 0.0002; spectrum_error_over_sigma1=0.0002341161510328827 > 0.0002
- `dense1024` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.0002111266301765788 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0005293208747631395 > 0.0002; reconstruction_relative_fro=0.00030949562120323004 > 0.0002
- `dense1024` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.02086268540871732 > 0.0002
- `dense1024` / `qr_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.00021247471655940644 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0005132035006172926 > 0.0002; reconstruction_relative_fro=0.0002604908019704124 > 0.0002
- `dense2048` / `jacobi`: timeout; Worker timeout; limit 60 seconds covers startup, warmup, timed repetitions and validation.
- `dense2048` / `gram_jacobi`: timeout; Worker timeout; limit 60 seconds covers startup, warmup, timed repetitions and validation.
- `dense2048` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.00044601276409111376 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0010666371316931944 > 0.0002; reconstruction_relative_fro=0.0006066366255313718 > 0.0002; spectrum_error_over_sigma1=0.00031289124378813663 > 0.0002
- `dense2048` / `native_Xgesvdp`: ok; reconstruction_relative_fro=53.948344526254736 > 0.0002; spectrum_error_over_sigma1=26.90633017834649 > 0.0002
- `dense2048` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.02569989762070608 > 0.0002; spectrum_error_over_sigma1=0.0003244992232661796 > 0.0002
- `tall8192x512` / `vendor_gesvdj`: ok; orthogonality_v_fro_over_sqrt_rows=0.0002239193699971151 > 0.0002
- `tall8192x512` / `qr_library`: ok; orthogonality_v_fro_over_sqrt_rows=0.0002239193699971151 > 0.0002
- `cluster512_fp32` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.009305799756431138 > 0.0002; reconstruction_relative_fro=0.007422607495251718 > 0.0002; spectrum_error_over_sigma1=0.007429639507005357 > 0.0002
- `cluster512_fp32` / `vendor_gesvdj`: ok; orthogonality_v_fro_over_sqrt_rows=0.0002883163648271271 > 0.0002
- `cluster512_fp32` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.06700368397414147 > 0.0002
- `cluster512_fp32` / `qr_library`: ok; orthogonality_v_fro_over_sqrt_rows=0.00024443461780788067 > 0.0002
- `cluster512_fp64` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.0037157698607964023 > 1e-09; reconstruction_relative_fro=1.7179376230500901e-07 > 1e-09; spectrum_error_over_sigma1=1.6819243253222363e-07 > 1e-09
- `cluster512_fp64` / `vendor_gesvda`: error; _LinAlgError: linalg.svd: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: 475).
- `cluster512_fp64` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=0.68247267236108 > 1e-09; reconstruction_relative_fro=8.037549739579065e-09 > 1e-09; spectrum_error_over_sigma1=1.3534345092962655e-08 > 1e-09
- `cluster1024_fp64` / `gram_jacobi`: ok; orthogonality_u_fro_over_sqrt_cols=0.000958217152806868 > 1e-09
- `cluster1024_fp64` / `gram_library`: ok; orthogonality_u_fro_over_sqrt_cols=5.995507984317501e-05 > 1e-09
- `lowrank2048x1024_k32` / `native_Xgesvdr`: ok; residual_over_optimal_tail=1.1529497629715546 > 1.05
- `lowrank2048x1024_k32` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.0003303996036161399 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.00027490320119766433 > 0.0002
- `lowrank8192x2048_k64` / `native_Xgesvdr`: ok; residual_over_optimal_tail=1.282353155466935 > 1.05
- `lowrank8192x2048_k64` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.0005826485703115507 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0004709139120613724 > 0.0002
- `dense1024_top32` / `randomized`: ok; resolved_spectrum_max_relative_error=0.07809252513109864 > 0.01
- `dense1024_top32` / `subspace`: ok; resolved_spectrum_max_relative_error=0.07809252513109864 > 0.01
- `dense1024_top32` / `block_krylov`: ok; resolved_spectrum_max_relative_error=0.03978508545145563 > 0.01
- `dense1024_top32` / `randomized_library`: ok; resolved_spectrum_max_relative_error=0.07656696747991779 > 0.01
- `dense1024_top32` / `randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.02627456895811794 > 0.01
- `dense1024_top32` / `native_Xgesvdr`: ok; resolved_spectrum_max_relative_error=0.07934012434179348 > 0.01
- `dense1024_top32` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.00029590271761912413 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0004844514647960505 > 0.0002
- `sparse2048x1024_d01` / `randomized`: ok; resolved_spectrum_max_relative_error=0.0887775862755671 > 0.01
- `sparse2048x1024_d01` / `block_krylov`: ok; resolved_spectrum_max_relative_error=0.04707743529149578 > 0.01
- `sparse2048x1024_d01` / `randomized_library`: ok; resolved_spectrum_max_relative_error=0.08712530149678929 > 0.01
- `sparse2048x1024_d01` / `randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.025986909850365898 > 0.01
- `sparse2048x1024_d01` / `sparse_randomized_library`: ok; resolved_spectrum_max_relative_error=0.08877636513149903 > 0.01
- `sparse2048x1024_d01` / `sparse_randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.030780918804734245 > 0.01
- `sparse2048x1024_d01` / `native_Xgesvdr`: ok; resolved_spectrum_max_relative_error=0.09130160215323398 > 0.01
- `sparse2048x1024_d01` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.0003074117785038307 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.00044858426905504623 > 0.0002
- `sparse4096x2048_d001` / `randomized`: ok; resolved_spectrum_max_relative_error=0.10171755131140182 > 0.01
- `sparse4096x2048_d001` / `block_krylov`: ok; resolved_spectrum_max_relative_error=0.05651989049216889 > 0.01
- `sparse4096x2048_d001` / `randomized_library`: ok; resolved_spectrum_max_relative_error=0.10591619677828815 > 0.01
- `sparse4096x2048_d001` / `randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.03515713580202377 > 0.01
- `sparse4096x2048_d001` / `sparse_randomized_library`: ok; resolved_spectrum_max_relative_error=0.10171632029961719 > 0.01
- `sparse4096x2048_d001` / `sparse_randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.03545224289984827 > 0.01
- `sparse4096x2048_d001` / `native_Xgesvdr`: ok; resolved_spectrum_max_relative_error=0.1719450390170305 > 0.01
- `sparse4096x2048_d001` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.0006102397005243841 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.0007410167892096813 > 0.0002
- `sparse4096x2048_d01` / `sparse_lanczos`: ok; resolved_spectrum_max_relative_error=0.013447746993479065 > 0.01
- `sparse4096x2048_d01` / `lanczos`: ok; resolved_spectrum_max_relative_error=0.013445937167990005 > 0.01
- `sparse4096x2048_d01` / `randomized`: ok; resolved_spectrum_max_relative_error=0.09968824759069406 > 0.01
- `sparse4096x2048_d01` / `block_krylov`: ok; resolved_spectrum_max_relative_error=0.05926051975195527 > 0.01
- `sparse4096x2048_d01` / `randomized_library`: ok; resolved_spectrum_max_relative_error=0.10282430616846398 > 0.01
- `sparse4096x2048_d01` / `randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.04440369620101557 > 0.01
- `sparse4096x2048_d01` / `sparse_randomized_library`: ok; resolved_spectrum_max_relative_error=0.09969115000492615 > 0.01
- `sparse4096x2048_d01` / `sparse_randomized_library_q4`: ok; resolved_spectrum_max_relative_error=0.042991157294322804 > 0.01
- `sparse4096x2048_d01` / `native_Xgesvdr`: ok; resolved_spectrum_max_relative_error=0.10517947583661003 > 0.01
- `sparse4096x2048_d01` / `vendor_gesvdj`: ok; orthogonality_u_fro_over_sqrt_cols=0.000390268536319128 > 0.0002; orthogonality_v_fro_over_sqrt_rows=0.000933669702164336 > 0.0002
- `sparse4096x2048_d001` / `native_Xgesvdr_q8`: ok; resolved_spectrum_max_relative_error=0.0107412480656717 > 0.01
- `sparse4096x2048_d01` / `sparse_randomized_library_q8`: ok; resolved_spectrum_max_relative_error=0.010478999752501782 > 0.01
- `lowrank2048x1024_k32` / `native_Xgesvdr_q4`: ok; residual_over_optimal_tail=1.152915393010128 > 1.05
- `lowrank2048x1024_k32` / `native_Xgesvdr_q8`: ok; residual_over_optimal_tail=1.152859087611474 > 1.05
- `lowrank8192x2048_k64` / `native_Xgesvdr_q4`: ok; residual_over_optimal_tail=1.2823218588540923 > 1.05
- `lowrank8192x2048_k64` / `native_Xgesvdr_q8`: ok; residual_over_optimal_tail=1.2822574083785945 > 1.05

## Measurement contract and limits

- Full-spectrum FP32 acceptance: reconstruction, spectrum error/sigma1 and normalized U/V orthogonality ≤2e-4; FP64: ≤1e-9. Sorted nonnegative singular values are required.
- Top-k acceptance: residual ≤1.05× the optimal reference tail, maximum relative error of the resolved leading k singular values ≤1%, and the same orthogonality limits. If the reference tail is zero, the full reconstruction tolerance replaces the tail-ratio check.
- The independent reference is SciPy LAPACK gesvd in FP64 on the actual rounded input. Values below its dimension-scaled reference floor are excluded from relative spectrum errors. The reference is not arbitrary precision.
- Timing uses resident GPU input, synchronized wall clocks and CUDA events. Generation, CPU reference, transfers/preparation and validation are excluded from solver medians. TF32 is disabled, four CPU threads are used, and no CUDA graphs are replayed.
- Memory is the Torch allocator peak during timed solves, including prepared input, outputs and Torch-visible workspace. It excludes CUDA context/driver overhead and allocations outside the allocator. Reserved and incremental peaks are also preserved in the raw records.
- Sparse methods use actual sparse COO products, but the harness also retains dense GPU input and dense CPU references. Their reported peaks do not establish the minimum memory needed by a pure sparse application. Sparse conversion is outside resident solve time.
- Xgesvdr uses the previously inspected legacy wrapper, which allocates full square U/V buffers even for a truncated result. Its memory result describes this wrapper, not an intrinsic requirement of randomized cuSOLVER.
- Explicit vendor top-k SVD controls compute a full decomposition and slice it. Randomized library controls use optimized QR/SVD primitives; they are not standalone decompositions. Native wrappers include allocation, workspace query and status synchronization inside the measured call.
- One input seed is used per matrix. Randomized implementations use different sketches; the native routine seed is not controlled through the legacy wrapper. Timing repeats are not independent accuracy trials. These results support the listed workloads, not a universal ranking.
- Matrices and references are preserved in the server experiment directory. The machine-readable summary retains the complete winning records; measurements.jsonl contains every attempt, including failures. Coarse device telemetry is observational, not per-call energy or occupancy.

![Speed and memory](broad-winners.png)
