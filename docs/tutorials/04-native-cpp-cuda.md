# Tutorial 4: C++ and CUDA

**What this tutorial is about.** You write an SVD, or one step of one, in C++ or with your
own CUDA kernel, and check it exactly like a Python method. You don't need CMake or a
`setup.py`: one source file and the same `python -m svdlab check` command. `svdlab`
compiles the file through PyTorch, caches the build, and calls your function with PyTorch
tensors.

**You will learn:**

- what needs to be installed, and how to ask `svdlab` what is missing;
- the C++ form of the function contract;
- three steps from a toolchain test to a hand-written CUDA kernel;
- how to call native code from Python and use a C++ function as a step of a recipe;
- how to fix the usual build problems.

**Before you start:** [tutorial 1](01-install-and-first-check.md) and the contract from
[tutorial 2](02-write-your-own-svd.md). Basic C++; CUDA only for section 4.6.

**Notebook version:** [notebooks/04_native_cpp_cuda.ipynb](../../notebooks/04_native_cpp_cuda.ipynb)
(cells that need a missing tool say so and skip).

## 4.1 How it works

`python -m svdlab check my_svd.cpp` compiles the file with PyTorch's
`torch.utils.cpp_extension.load`, imports the resulting Python module and calls its `svd`
function with the same arguments as a Python method. Your C++ code receives a
`torch::Tensor`, so it can use the whole ATen tensor library (the C++ side of PyTorch),
raw pointers, or your own CUDA kernels.

- `.cpp` and `.cc` files are compiled with the host C++ compiler.
- `.cu` files are compiled with `nvcc` (CUDA).
- The build is cached in `~/.cache/torch_extensions` (or `$TORCH_EXTENSIONS_DIR`). The cache
  key contains a hash of the file, so the first run takes 30 to 60 seconds and later runs
  start immediately until you change the file.

## 4.2 Requirements

| What | Install | Needed for |
| --- | --- | --- |
| ninja | `pip install ninja` or `pip install -e '.[native]'` | `.cpp`, `.cu` |
| Python headers (`Python.h`) | Debian/Ubuntu: `sudo apt install python3-dev` (or `python3.12-dev` for a specific version); macOS: included with python.org and Homebrew Python | `.cpp`, `.cu` |
| A C++17 compiler | `g++` or `clang++` | `.cpp`, `.cu` |
| CUDA toolkit with `nvcc`, matching the CUDA version of your PyTorch build | NVIDIA's installer; set `CUDA_HOME` if it is not in `/usr/local/cuda` | `.cu` only |

You can ask `svdlab` what is missing:

```python
from svdlab import native
print(native.missing_tools())            # for .cpp; [] means ready
print(native.missing_tools(cuda=True))   # for .cu
```

When something is missing, `python -m svdlab check` lists it and exits with code 2:

```text
Cannot load examples/native/my_svd.cu: RuntimeError: Cannot build my_svd.cu; missing:
  - nvcc; install the CUDA toolkit or set CUDA_HOME (e.g. /usr/local/cuda)
```

## 4.3 The contract in C++

```cpp
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(torch::Tensor a, int64_t rank) {
  ...
  return {u, s, vh};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("svd", &svd); }
```

The rules are those of tutorial 2: `a` is `(m, n)` on the tested device (contiguous;
`svdlab` makes sure of that), return `U (m, k)`, `s (k,)` descending, `Vh (k, n)` with
`k >= rank`, on the same device as `a`. `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` is
required; `TORCH_EXTENSION_NAME` is filled in by the build.

## 4.4 Step 1: check the toolchain

[examples/native/minimal.cpp](../../examples/native/minimal.cpp) just calls the library SVD,
so every case passes if the build works:

```cpp
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(torch::Tensor a, int64_t rank) {
  auto [u, s, vh] = torch::linalg_svd(a, /*full_matrices=*/false);
  const int64_t k = std::min<int64_t>(rank, s.size(0));
  return {u.narrow(1, 0, k), s.narrow(0, 0, k), vh.narrow(0, 0, k)};
}
```

```bash
python -m svdlab check examples/native/minimal.cpp --quick
```

```text
minimal.cpp: 6/6 ok, total median 1.272 ms
```

If this fails, fix the build before writing your own code; section 4.8 lists the usual
causes.

## 4.5 Step 2: a real algorithm with ATen

[examples/native/my_svd.cpp](../../examples/native/my_svd.cpp) is the one-sided Jacobi SVD
of `examples/my_svd.py`, translated line by line into ATen calls (`index_select`,
`index_copy_`, `torch::where`, ...). Because it uses only tensor operations, the same
compiled code runs on CPU and on GPU tensors, real and complex:

```bash
python -m svdlab check examples/native/my_svd.cpp --quick --compare examples/my_svd.py
python -m svdlab check examples/native/my_svd.cpp --device cuda      # on a GPU host
```

```text
my_svd.cpp: 6/6 ok, total median 259.152 ms
my_svd: 6/6 ok, total median 558.045 ms
```

The errors agree with the Python version to two or three digits and it is about twice as fast, because the
Python interpreter overhead per operation is gone. The tensor operations themselves are
the same.

This is the recommended starting point for C++: translate a working Python prototype
into ATen first, check that the numbers match, then optimize the hot parts.

## 4.6 Step 3: your own CUDA kernel

[examples/native/my_svd.cu](../../examples/native/my_svd.cu) implements the same Jacobi
sweeps with a hand-written kernel: one thread block per column pair computes the three
inner products with a shared-memory reduction and applies the rotation to the columns of
W and V. The host code loops over rounds and sweeps and launches the kernel on PyTorch's
current CUDA stream:

```cpp
auto stream = at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "jacobi_svd", [&] {        // float and double
  rotate_pairs<scalar_t><<<pairs, THREADS, 0, stream>>>(w.data_ptr<scalar_t>(), ...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                       // reports launch errors
});
```

```bash
python -m svdlab check examples/native/my_svd.cu --device cuda --compare torch
python -m svdlab check examples/native/my_svd.cu --gpu-check
```

The example supports real float32 and float64 input only; on the complex case it raises
an error through `TORCH_CHECK`, which the report shows as `ERROR` for that case. That is
expected and shows how unsupported input is reported. Things to keep in mind when you
write kernels:

- Use `TORCH_CHECK(condition, "message")` for input checks; the message reaches the report.
- Launch on `at::cuda::getCurrentCUDAStream()` so that the timing (which synchronizes the
  device) measures your kernels.
- Use `AT_DISPATCH_FLOATING_TYPES` (or `..._AND_COMPLEX_TYPES`) to support several dtypes.
- Keep the result on the GPU. Returning CPU tensors fails `--gpu-check`.

## 4.7 Native code from Python, and native blocks

`svdlab.native.load` returns the compiled `svd` as a Python function:

```python
from svdlab import native, check, QUICK_SUITE
svd = native.load('examples/native/my_svd.cpp')
rows = check(svd, QUICK_SUITE)
```

A file can export more than `svd`. `native.build` returns the whole compiled module, so a
C++ building block can replace a step of a Python recipe (tutorial 3).
`examples/native/minimal.cpp` also exports `qr` with the contract of `svdlab.blocks.qr`:

```python
from svdlab import native, recipes
module = native.build('examples/native/minimal.cpp')

def svd(a, rank):
    return recipes.randomized(a, rank, qr=module.qr)    # C++ QR inside the Python recipe
```

Use `native.build(path, verbose=True)` to see the compiler command lines and messages.

## 4.8 Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Python headers ... Python.h` in the missing list | `sudo apt install python3-dev` (the version must match the Python of your environment) |
| `ninja` in the missing list | `pip install ninja` in the same environment |
| `nvcc` in the missing list | install the CUDA toolkit, or `export CUDA_HOME=/usr/local/cuda-12.x` |
| `nvcc` version does not match the PyTorch CUDA version | use the toolkit that matches `torch.version.cuda` |
| A compiler error | it is printed in full; run `native.build(path, verbose=True)` for the commands |
| `does not export 'svd'` | add `m.def("svd", &svd);` inside `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` |
| A change is not picked up | the cache key is the file hash, so save the file; headers you `#include` are not hashed: touch the `.cpp` or clear `~/.cache/torch_extensions` |
| `output_device` is `cpu` on a GPU run | the code moved data to the host (for example `.cpu()` or a CPU `torch::zeros` without `a.options()`) |

## Next

- [Tutorial 5](05-metrics-and-matrices.md): every metric explained, your own matrices,
  saving results, the GPU check in detail.
