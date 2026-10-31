"""Building blocks of SVD algorithms, for one matrix (no batch dimension).

The blocks are the repository's own implementations (`svdbench.standalone`): tensor
arithmetic and matrix products only, no vendor factorization. They run on CPU and GPU,
real and complex. `torch_*` versions call the library and are useful as a control.

    from svdlab import blocks
    q, r = blocks.qr(a)                    # replace with your own QR, keep the rest
    q, d, e, p = blocks.bidiagonalize(a)   # a = q @ B @ p^H, B bidiagonal(d, e)
    u, d, e, v = blocks.lanczos_bidiagonalize(a, steps=20)   # first 20 columns of the same, via products
    u, s, v = blocks.bidiagonal_svd(d, e)  # B = u @ diag(s) @ v^T

`svdlab.recipes` assembles these into complete SVDs whose steps you can swap.
"""
import torch
from svdbench.standalone import primitives as P
from svdbench.standalone import svd as _standalone_svd
from svdbench.standalone.iterative import DenseOperator, lanczos_bidiagonalize as _lanczos
from svdbench.standalone.solvers import ALGORITHMS


def _tall(a, name):
    if a.ndim != 2 or a.shape[0] < a.shape[1]:
        raise ValueError(f'{name} expects one tall matrix (m >= n); got shape {tuple(a.shape)}. '
                         'For a wide matrix use a.mH and swap the factors.')


def qr(a):
    """Householder QR of a tall matrix: Q (m, n) orthonormal columns, R (n, n) upper triangular."""
    _tall(a, 'qr')
    q, r = P.qr(a[None])
    return q[0], r[0]


def torch_qr(a):
    """Library QR (LAPACK/cuSOLVER), same output as `qr`."""
    return torch.linalg.qr(a, mode='reduced')


def bidiagonalize(a):
    """Golub–Kahan bidiagonalization of a tall matrix: a = Q B Pᴴ.

    Returns Q (m, n), d (n,), e (n-1,), P (n, n). B has the real diagonal d and the
    real superdiagonal e.
    """
    _tall(a, 'bidiagonalize')
    d, e, q, p = P.bidiag(a[None])
    return q[0], d[0], e[0], p[0]


def bidiagonal_svd(d, e, max_iter=None, tol=None):
    """Implicit-shift QR (bdsqr) on the bidiagonal B: returns U, s (descending), V with B = U diag(s) Vᵀ."""
    s, u, v = P.bdsqr(d[None], e[None], max_iter=max_iter, tol=tol)
    return u[0], s[0], v[0]


def jacobi_svd(a, rank=None, max_sweeps=60, tol=None):
    """One-sided Jacobi SVD of any matrix: U (m, k), s (k,), Vh (k, n), k = rank or min(m, n)."""
    return svd(a, 'jacobi', rank, max_sweeps=max_sweeps, tol=tol)


def lanczos_bidiagonalize(a, steps, seed=0):
    """Golub–Kahan–Lanczos with full reorthogonalization, from a random start vector.

    Returns U (m, k), alpha (k,), beta (k-1,), V (n, k) with k = steps and Uᴴ a V = B,
    B upper bidiagonal (real diagonal alpha, superdiagonal beta). Unlike `bidiagonalize`
    it touches a only through products a @ x and aᴴ @ y, and its leading singular values
    converge first, so a few steps approximate the top of the spectrum.
    """
    scale = a.abs().max().clamp_min(torch.finfo(a.real.dtype).tiny)
    u, alpha, beta, v, _ = _lanczos(DenseOperator(a / scale), steps, seed)
    return u, alpha * scale, beta * scale, v


def tridiagonalize(h):
    """Householder reduction of a Hermitian matrix: h = Q T Qᴴ. Returns Q, d (n,), e (n-1,)."""
    d, e, q = P.tridiag(h[None])
    return q[0], d[0], e[0]


def hermitian_eig(h, max_sweeps=30, tol=None):
    """Jacobi eigendecomposition of a Hermitian matrix: w (n,) ascending, V (n, n)."""
    w, v, _ = P.jacobi_eigh(h[None], tol=tol, max_sweeps=max_sweeps)
    return w[0], v[0]


def cholesky(g):
    """L lower triangular with g = L Lᴴ; raises ValueError if g is not positive definite."""
    l, fail = P.cholesky(g[None])
    if bool(fail.any()):
        raise ValueError('cholesky: matrix is not positive definite in working precision')
    return l[0]


def solve_lower(l, b):
    """X with L X = B, L lower triangular (n, n), B (n, c)."""
    return P.trsm_lower(l[None], b[None])[0]


def solve_upper(u, b):
    """X with U X = B, U upper triangular (n, n), B (n, c)."""
    return P.trsm_upper(u[None], b[None])[0]


def random_sketch(a, columns, seed=0):
    """Gaussian test matrix (n, columns) with a's dtype and device, reproducible by seed."""
    gen = torch.Generator(device=a.device).manual_seed(seed)
    return torch.randn(a.shape[-1], columns, dtype=a.dtype, device=a.device, generator=gen)


def complete_basis(q, size):
    """Extend orthonormal columns q (m, k) to `size` orthonormal columns."""
    return P.complete_basis(q[None], size)[0]


def svd(a, algorithm='jacobi', rank=None, **options):
    """Any of the 16 repository SVDs (`svdlab.blocks.ALGORITHMS`): U (m, k), s (k,), Vh (k, n)."""
    if rank is not None:
        options['rank'] = rank
    options = {k: v for k, v in options.items() if v is not None}
    u, s, vh = _standalone_svd(a[None], algorithm=algorithm, **options)
    return u[0], s[0], vh[0]


def torch_svd(a, rank=None):
    """Library SVD (LAPACK/cuSOLVER), truncated to rank."""
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    k = rank or len(s)
    return u[:, :k], s[:k], vh[:k]


__all__ = ['ALGORITHMS', 'qr', 'torch_qr', 'bidiagonalize', 'lanczos_bidiagonalize', 'bidiagonal_svd', 'jacobi_svd', 'tridiagonalize',
           'hermitian_eig', 'cholesky', 'solve_lower', 'solve_upper', 'random_sketch', 'complete_basis',
           'svd', 'torch_svd']
