"""Bridge to the user's inspected Hetzner binding snapshot, kept unmodified.

This bridge includes allocation/query/copy overhead in every measured call.
It is not a preallocated kernel-only benchmark. Large full output buffers in
the legacy randomized wrapper are bounded here, not silently misattributed
to the cuSOLVER workspace requirement.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

from .algorithms import Unsupported

_module = None


def prepare(a, spec):
    global _module
    if a.device.type != "cuda" or torch.version.hip:
        raise Unsupported("The legacy cuSOLVER bridge requires NVIDIA CUDA")
    if spec.get("mode", "reduced") != "reduced":
        raise Unsupported("Legacy bridge exposes reduced output only")
    path = Path(__file__).resolve().parent.parent / "references/hetzner-cusolver_direct.py"
    if not path.exists():
        raise Unsupported("Run from the source checkout containing the inspected reference binding")
    if _module is None:
        loader = importlib.util.spec_from_file_location("svd_research_legacy_cusolver", path)
        _module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(_module)
    routine = spec["routine"]
    x = a.unsqueeze(0) if a.ndim == 2 else a
    b, m, n = x.shape
    if routine == "gesvdjBatched" and max(m, n) > 32:
        raise Unsupported("gesvdjBatched requires both m and n <= 32")
    if routine == "gesvdaStridedBatched" and m < n:
        raise Unsupported("Legacy strided adapter requires m >= n")
    if routine == "Xgesvdr":
        k, p = spec.get("rank", 8), spec.get("oversample", 8)
        if k + p > min(m, n):
            raise Unsupported("Xgesvdr requires rank + oversample <= min(m,n)")
        output_capacity = b * (m*m + n*n) * a.element_size()
        if output_capacity > 512 * 1024**2:
            raise Unsupported("Legacy wrapper allocates full U/V; >512 MiB guard. This is not a cuSOLVER workspace limit.")
        fn = lambda: _module.gesvdr(x, k, p, spec.get("iterations", 2))
    elif routine == "Xgesvdp":
        fn = lambda: _module.gesvdp(x)
    elif routine == "gesvdjBatched":
        fn = lambda: _module.gesvdj_batched(x, max_sweeps=spec.get("max_sweeps", 100))
    elif routine == "gesvdaStridedBatched":
        fn = lambda: _module.gesvda_strided(x, rank=spec.get("rank"))
    else:
        raise Unsupported(f"Unknown native routine: {routine}")

    def solve():
        out = fn()
        if out.get("info_nonzero", 0):
            raise RuntimeError(f"{routine}: {out['info_nonzero']} nonzero convergence statuses")
        solve.diagnostics = {"info_nonzero": out.get("info_nonzero", 0)}
        if "h_err_sigma" in out:
            solve.diagnostics["h_err_sigma"] = out["h_err_sigma"].tolist()
        result = out["U"], out["s"], out["V"].transpose(-2, -1).conj()
        return tuple(t[0] for t in result) if a.ndim == 2 else result

    solve.metadata = {"cusolver_version": _module.version(),
                      "binding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "library_path": _module.lib()._name,
                      "contract": "Legacy wrapper: allocation, workspace query, device input copy and info synchronization included; batched Xgesvd* use a Python loop."}
    return solve
