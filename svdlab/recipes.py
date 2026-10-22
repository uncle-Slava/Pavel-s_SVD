"""Complete SVDs assembled from `svdlab.blocks`, with every step replaceable.

Each recipe has the scorecard signature `(a, rank) -> (U, s, Vh)`; pass your own
function for any step as a keyword argument:

    from svdlab import blocks, recipes

    def my_qr(y):                          # same contract as blocks.qr
        ...
        return q, r

    def svd(a, rank):
        return recipes.randomized(a, rank, qr=my_qr)

Then `python -m svdlab check this_file.py` checks it. Wide matrices are handled by
factorizing the adjoint, so the steps only ever see tall inputs.
"""
import torch
from . import blocks


def _adjoint_if_wide(recipe, a, rank, **steps):
    if a.shape[0] < a.shape[1]:
        u, s, vh = recipe(a.mH, rank, **steps)
        return vh.mH, s, u.mH
    return None


def golub_kahan(a, rank=None, *, bidiagonalize=blocks.bidiagonalize, bidiagonal_svd=blocks.bidiagonal_svd):
    """a = Q B Pᴴ (bidiagonalize), B = U diag(s) Vᵀ (bidiagonal_svd), then U_a = Q U, V_a = P V."""
    flipped = _adjoint_if_wide(golub_kahan, a, rank, bidiagonalize=bidiagonalize, bidiagonal_svd=bidiagonal_svd)
    if flipped:
        return flipped
    scale = a.abs().max().clamp_min(torch.finfo(a.real.dtype).tiny)
    q, d, e, p = bidiagonalize(a / scale)
    ub, s, vb = bidiagonal_svd(d, e)
    k = rank or len(s)
    u, v = q @ ub.to(a.dtype), p @ vb.to(a.dtype)
    return u[:, :k], s[:k] * scale, v[:, :k].mH


def qr_then_svd(a, rank=None, *, qr=blocks.qr, small_svd=blocks.jacobi_svd):
    """a = Q R (qr), R = U s Vh (small_svd on n x n), then U_a = Q U."""
    flipped = _adjoint_if_wide(qr_then_svd, a, rank, qr=qr, small_svd=small_svd)
    if flipped:
        return flipped
    q, r = qr(a)
    u, s, vh = small_svd(r)
    k = rank or len(s)
    return (q @ u)[:, :k], s[:k], vh[:k]


def randomized(a, rank, *, oversample=8, power_iterations=2, seed=0,
               sketch=blocks.random_sketch, qr=blocks.qr, small_svd=blocks.jacobi_svd):
    """Halko–Martinsson–Tropp: Q = orth(A Ω), power iterations with QR each half step,
    then the small SVD of Qᴴ A. Accurate when the spectrum decays past `rank`."""
    columns = min(rank + oversample, *a.shape)
    q, _ = qr(a @ sketch(a, columns, seed))
    for _ in range(power_iterations):
        z, _ = qr(a.mH @ q)
        q, _ = qr(a @ z)
    u, s, vh = small_svd(q.mH @ a)
    return (q @ u)[:, :rank], s[:rank], vh[:rank]


def jacobi(a, rank=None):
    """One-sided Jacobi on the whole matrix: slow but accurate, a good reference."""
    return blocks.jacobi_svd(a, rank)


__all__ = ['golub_kahan', 'qr_then_svd', 'randomized', 'jacobi']
