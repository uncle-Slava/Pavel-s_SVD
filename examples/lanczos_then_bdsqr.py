"""Lanczos bidiagonalization (only products with A), then your own bidiagonal SVD.

    python -m svdlab check examples/lanczos_then_bdsqr.py --quick --device cuda

Here the small bidiagonal B = diag(d) + superdiag(e) is solved with the library SVD;
replace my_bdsqr with your own implicit-shift QR, divide and conquer, Jacobi, ...
"""
import torch
from svdlab import recipes


def my_bdsqr(d, e):
    """Same contract as svdlab.blocks.bidiagonal_svd: B = U diag(s) Vᵀ, s descending."""
    u, s, vh = torch.linalg.svd(torch.diag(d) + torch.diag(e, 1))
    return u, s, vh.mT


def svd(a, rank):
    return recipes.lanczos(a, rank, bidiagonal_svd=my_bdsqr)
