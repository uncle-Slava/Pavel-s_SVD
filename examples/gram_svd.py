"""Tutorial example: a fast SVD that fails on hard matrices, and a repaired variant.

    python -m svdlab check examples/gram_svd.py --quick                 # svd: 4/6 ok
    python -m svdlab check examples/gram_svd.py:refined --quick         # refined: 6/6 ok

`svd` takes the eigendecomposition of the Gram matrix AᴴA. It is short and fast, but
forming AᴴA squares the condition number, so singular values below about
sqrt(eps)·σ₁ are lost and U loses orthogonality on ill-conditioned or clustered spectra.
`refined` keeps the Gram eigenvectors only as a starting basis and lets one-sided Jacobi
finish on A V, whose columns are already nearly orthogonal, so it needs few sweeps.
See docs/tutorials/02-write-your-own-svd.md.
"""
import torch
from svdlab import blocks


def svd(a, rank):
    """SVD through the eigendecomposition of the Gram matrix AᴴA."""
    g = a.mH @ a                                   # (n, n) Hermitian, eigenvalues = s**2
    w, v = torch.linalg.eigh(g)                    # ascending
    w, v = w.flip(0), v.flip(1)                    # descending
    s = w.clamp_min(0).sqrt()
    u = (a @ v) / s.clamp_min(torch.finfo(s.dtype).tiny)
    return u[:, :rank], s[:rank], v[:, :rank].mH


def refined(a, rank):
    """Gram eigenvectors as a starting basis, then Jacobi on A V to repair the accuracy."""
    wide = a.shape[0] < a.shape[1]
    x = a.mH if wide else a                        # work on a tall matrix
    _, v = torch.linalg.eigh(x.mH @ x)
    u, s, wh = blocks.jacobi_svd(x @ v)            # x V = U diag(s) Wᴴ, so x = U diag(s) (V W)ᴴ
    vh = wh @ v.mH
    u, s, vh = u[:, :rank], s[:rank], vh[:rank]
    return (vh.mH, s, u.mH) if wide else (u, s, vh)
