"""Template: replace one step of an existing SVD and keep the rest.

Here the QR inside randomized SVD is replaced by CholeskyQR2. Run both, side by side:

    python -m svdlab check examples/replace_a_step.py --compare svdlab.recipes:randomized

Expect two failures: CholeskyQR forms YᴴY, which squares the condition number, so it breaks
on ill_conditioned_1e8 and on the float32 low-rank case while Householder QR does not.
Catching exactly this kind of trade-off is what the check is for.

Other steps you can replace the same way (see svdlab/recipes.py):
    recipes.randomized(a, rank, sketch=..., qr=..., small_svd=..., power_iterations=...)
    recipes.golub_kahan(a, rank, bidiagonalize=..., bidiagonal_svd=...)
    recipes.qr_then_svd(a, rank, qr=..., small_svd=...)
Ready-made steps are in svdlab.blocks (qr, bidiagonalize, bidiagonal_svd, jacobi_svd, ...).
"""
import torch
from svdlab import recipes


def cholesky_qr2(y):
    """Same contract as svdlab.blocks.qr: tall y (m, n) -> Q (m, n) orthonormal, R (n, n) upper."""
    q, r = y, torch.eye(y.shape[1], dtype=y.dtype, device=y.device)
    for _ in range(2):                            # the second pass repairs the lost orthogonality
        l = torch.linalg.cholesky(q.mH @ q)       # QᴴQ = L Lᴴ
        q = torch.linalg.solve_triangular(l.mH, q, upper=True, left=False)   # Q L⁻ᴴ
        r = l.mH @ r
    return q, r


def svd(a, rank):
    return recipes.randomized(a, rank, qr=cholesky_qr2)
