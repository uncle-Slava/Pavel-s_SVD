"""Portable algorithm bodies. Native extensions can implement the same contract.

Input is a device tensor; output is (U, s, Vh). Never transfer to CPU implicitly.
RNG and sketch transfer are preparation, outside solver-only timings.
"""
import importlib

import numpy as np
import torch


class Unsupported(RuntimeError):
    pass


def adjoint(a):
    return a.transpose(-2, -1).conj()


def prepare(a, spec, seed):
    name = spec["name"]
    mode = spec.get("mode", "reduced")
    if mode not in ("reduced", "full", "values"):
        raise ValueError("mode must be reduced, full or values")
    rank = spec.get("rank", min(a.shape[-2:]))
    if not 1 <= rank <= min(a.shape[-2:]):
        raise ValueError("rank must be between 1 and min(m,n)")
    if name in ("randomized", "gram") and mode != "reduced":
        raise Unsupported(f"{name} implements reduced output only")
    if name == "randomized":
        p, q = spec.get("oversample", 16), spec.get("iterations", 2)
        if p < 0 or q < 0:
            raise ValueError("oversample and iterations must be nonnegative")
        ell = min(rank + p, min(a.shape[-2:]))
        rng = np.random.default_rng(seed)
        shape = (*a.shape[:-2], a.shape[-1], ell)
        omega = rng.standard_normal(shape)
        if a.is_complex():
            omega = (omega + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        omega = torch.as_tensor(omega, dtype=a.dtype, device=a.device)

        def randomized():
            basis = torch.linalg.qr(a @ omega, mode="reduced")[0]
            for _ in range(q):
                z = torch.linalg.qr(adjoint(a) @ basis, mode="reduced")[0]
                basis = torch.linalg.qr(a @ z, mode="reduced")[0]
            u, s, vh = torch.linalg.svd(adjoint(basis) @ a, full_matrices=False)
            return (basis @ u[..., :rank], s[..., :rank], vh[..., :rank, :])

        return randomized
    if name == "gram":
        # Intentionally expose loss of accuracy from squaring the condition number.
        # Zero singular values produce zero U/V columns, not valid null-space bases.
        def gram():
            tall = a.shape[-2] >= a.shape[-1]
            x = a if tall else adjoint(a)
            w, v = torch.linalg.eigh(adjoint(x) @ x)
            s = w.flip(-1)[..., :rank].clamp(min=0).sqrt()
            v = v.flip(-1)[..., :rank]
            denom = torch.where(s > 0, s, torch.ones_like(s))
            u = (x @ v) / denom.unsqueeze(-2)
            u = torch.where((s > 0).unsqueeze(-2), u, torch.zeros_like(u))
            return (u, s, adjoint(v)) if tall else (v, s, adjoint(u))

        return gram
    if name == "qr_svd":
        if mode == "full":
            raise Unsupported("qr_svd does not complete rectangular null-space bases")

        def qr_svd():
            tall = a.shape[-2] >= a.shape[-1]
            x = a if tall else adjoint(a)
            q, r = torch.linalg.qr(x, mode="reduced")
            if mode == "values":
                return None, torch.linalg.svdvals(r), None
            u, s, vh = torch.linalg.svd(r, full_matrices=False)
            u = q @ u
            return (u, s, vh) if tall else (adjoint(vh), s, adjoint(u))

        return qr_svd
    if name in ("default", "gesvd", "gesvdj", "gesvda"):
        kw = {}
        if name != "default":
            if a.device.type != "cuda" or torch.version.hip:
                raise Unsupported("Explicit cuSOLVER drivers require NVIDIA CUDA")
            kw["driver"] = name
        if mode == "values":
            return lambda: (None, torch.linalg.svdvals(a, **kw), None)
        return lambda: torch.linalg.svd(a, full_matrices=(mode == "full"), **kw)
    if ":" in name:
        module, factory = name.split(":", 1)
        # A custom factory may query workspace and allocate reusable native buffers.
        return getattr(importlib.import_module(module), factory)(a, spec)
    raise Unsupported(f"Unknown algorithm: {name}")
