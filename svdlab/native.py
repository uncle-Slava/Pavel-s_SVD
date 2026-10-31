"""Compile one C++ or CUDA file with torch.utils.cpp_extension and use it as an SVD method.

The file defines

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(torch::Tensor a, int64_t rank);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("svd", &svd); }

A `.cu` file is compiled with nvcc, a `.cpp`/`.cc` file with the host C++ compiler.
The build is cached by torch (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions)
and is redone only when the file changes. Requirements: ninja (`pip install ninja`),
the Python headers (Debian/Ubuntu: `python3-dev`), a C++ compiler and, for `.cu`, nvcc.
"""
import hashlib
import os
from pathlib import Path
import shutil
import sysconfig


def _ninja_on_path():
    """pip's ninja lives in the environment's bin/, which is not on PATH when python is called directly."""
    try:
        import ninja
    except ImportError:
        return
    if not shutil.which('ninja'):
        os.environ['PATH'] = ninja.BIN_DIR + os.pathsep + os.environ.get('PATH', '')


def missing_tools(cuda=False):
    """Human-readable list of what is missing for a build; empty when everything is there."""
    missing = []
    include = sysconfig.get_paths()['include']
    extra = [d for var in ('CPATH', 'CPLUS_INCLUDE_PATH') for d in os.environ.get(var, '').split(os.pathsep) if d]
    if not any((Path(d) / 'Python.h').is_file() for d in [include, *extra]):
        missing.append(f'Python headers ({include}/Python.h); Debian/Ubuntu: sudo apt install python3-dev')
    _ninja_on_path()
    if not shutil.which('ninja'):
        missing.append('ninja; pip install ninja')
    if not (shutil.which(os.environ.get('CXX', 'c++')) or shutil.which('g++')):
        missing.append('a C++ compiler (g++ or clang++)')
    if cuda:
        from torch.utils.cpp_extension import CUDA_HOME
        if not CUDA_HOME or not (Path(CUDA_HOME) / 'bin' / 'nvcc').is_file():
            missing.append('nvcc; install the CUDA toolkit or set CUDA_HOME (e.g. /usr/local/cuda)')
    return missing


def build(path, verbose=False, extra_cflags=('-O2',), extra_cuda_cflags=('-O2',)):
    """Compile and import the extension module for `path`."""
    from torch.utils.cpp_extension import load
    path = Path(path).resolve()
    cuda = path.suffix == '.cu'
    _ninja_on_path()
    missing = missing_tools(cuda)
    if missing:
        raise RuntimeError(f'Cannot build {path.name}; missing:\n  - ' + '\n  - '.join(missing))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    name = f'svdlab_{"".join(c if c.isalnum() else "_" for c in path.stem)}_{digest}'
    return load(name=name, sources=[str(path)], extra_cflags=list(extra_cflags),
                extra_cuda_cflags=list(extra_cuda_cflags) if cuda else None, verbose=verbose)


def load(path, function='svd', verbose=False):
    """The compiled `function(a, rank)` from a C++/CUDA file, ready for svdlab.check()."""
    module = build(path, verbose=verbose)
    if not hasattr(module, function):
        raise AttributeError(f'{path} does not export {function!r}; add m.def("{function}", &{function}) '
                             'inside PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)')
    native = getattr(module, function)

    def svd(a, rank):
        return native(a.contiguous(), int(rank))
    svd.__name__ = Path(path).stem
    return svd
