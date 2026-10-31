"""Matrix-free Golub--Kahan bidiagonalization and small-core SVD updates."""
import torch
from . import primitives as P


class DenseOperator:
    def __init__(self, a):
        if a.ndim != 2:
            raise ValueError('DenseOperator requires a matrix')
        self.a, self.shape, self.dtype, self.device = a, a.shape, a.dtype, a.device

    def matmat(self, x):
        return self.a @ x

    def rmatmat(self, x):
        return self.a.mH @ x


def _reorth(x, basis):
    for _ in range(2):
        x = x - basis @ (basis.mH @ x)
    return x


def _extend(q):
    return P.complete_basis(q[None], q.shape[-1] + 1)[0, :, -1:]


def lanczos_bidiagonalize(operator, steps, seed=20261011):
    """`steps` Golub--Kahan--Lanczos steps with full reorthogonalization.

    Returns ub (m, k), alpha (k,), beta (k-1,), vb (n, k), info with k = steps and
    ub^H A vb = B, B upper bidiagonal with diagonal alpha and superdiagonal beta.
    A breakdown (alpha or beta below eps * max(m, n) * max|A v|) restarts with a new
    orthogonal direction. info['next_v'] is the following right vector (None when
    vb already spans all n columns); lanczos_svd uses it to widen its projection.
    """
    m, n = operator.shape
    if not 1 <= steps <= min(m, n):
        raise ValueError('steps must be in [1, min(m, n)]')
    kwargs = dict(dtype=operator.dtype, device=operator.device)
    gen = torch.Generator(device=operator.device).manual_seed(seed)
    v = torch.randn(n, 1, generator=gen, **kwargs)
    v = v / (v.abs() ** 2).sum().sqrt()
    ub, vb = torch.empty(m, 0, **kwargs), torch.empty(n, 0, **kwargs)
    prev = torch.zeros(m, 1, **kwargs)
    beta, restarts = 0., 0
    alphas, betas = [], []
    magnitude = torch.zeros((), dtype=v.real.dtype, device=v.device)
    for _ in range(steps):
        vb = torch.cat([vb, v], 1)
        av = operator.matmat(v)
        magnitude = torch.maximum(magnitude, av.abs().max())
        w = _reorth(av - beta * prev, ub)
        alpha = (w.abs() ** 2).sum().sqrt()
        threshold = P._eps(v.dtype) * max(m, n) * magnitude
        if bool(alpha <= threshold):
            u = _extend(ub); restarts += 1
        else:
            u = w / alpha
        ub = torch.cat([ub, u], 1)
        alphas.append(alpha)
        z = _reorth(operator.rmatmat(u) - alpha * v, vb)
        beta = (z.abs() ** 2).sum().sqrt()
        if vb.shape[-1] < n:
            if bool(beta <= threshold):
                v = _extend(vb); restarts += 1
                beta = 0.
            else:
                v = z / beta
        betas.append(torch.as_tensor(beta, dtype=alpha.dtype, device=alpha.device))
        prev = u
    info = {'steps': steps, 'breakdown_restarts': restarts, 'matmat_calls': steps, 'rmatmat_calls': steps,
            'next_v': v if vb.shape[-1] < n else None}
    return ub, torch.stack(alphas), torch.stack(betas[:-1]) if steps > 1 else alphas[0].new_zeros(0), vb, info


def lanczos_svd(operator, rank, steps=None, seed=20261011):
    """Full-reorthogonalized Golub--Kahan with breakdown restart.

    operator supplies shape, dtype, device, matmat(X), rmatmat(X). Sparse
    operators can implement those products without constructing a dense matrix.
    Returns U,s,V (not Vh), info. Small projected solve is standalone Jacobi.
    """
    from .solvers import svd
    m, n = operator.shape
    if not 1 <= rank <= min(m, n):
        raise ValueError('invalid rank')
    steps = min(m, n, max(rank, 2 * rank + 8) if steps is None else steps)
    if steps < rank:
        raise ValueError('steps must be >= rank')
    ub, _, _, vb, bidiag_info = lanczos_bidiagonalize(operator, steps, seed)
    if bidiag_info['next_v'] is not None:
        vb = torch.cat([vb, bidiag_info['next_v']], 1)
    # Recompute the small projection to account for finite-precision reorthogonalization.
    core = ub.mH @ operator.matmat(vb)
    uc, s, vhc = svd(core, 'jacobi')
    return (ub @ uc[:, :rank], s[:rank], vb @ vhc.mH[:, :rank],
            {'steps': steps, 'breakdown_restarts': bidiag_info['breakdown_restarts'],
             'matmat_calls': steps + 1, 'rmatmat_calls': steps,
             'projected_shape': list(core.shape)})


def rank_one_update(u, s, vh, left, right, rank=None):
    """Update U diag(s) Vh by left @ right^H through an augmented small core.

    The input factorization can already be truncated: the update is exact for
    that represented matrix; previous truncation error is not recoverable.
    """
    from .solvers import svd
    if u.ndim != 2 or vh.ndim != 2 or len(s) != u.shape[-1] or len(s) != vh.shape[0]:
        raise ValueError('inconsistent single-matrix reduced SVD')
    v = vh.mH
    left, right = left.reshape(-1, 1), right.reshape(-1, 1)
    if left.shape[0] != u.shape[0] or right.shape[0] != v.shape[0]:
        raise ValueError('update dimensions disagree')
    def augment(q, x):
        res = _reorth(x, q)
        nr = (res.abs() ** 2).sum().sqrt()
        nx = (x.abs() ** 2).sum().sqrt()
        if q.shape[1] < q.shape[0] and bool(nr > P._eps(q.dtype) * q.shape[0] * nx):
            q = torch.cat([q, res / nr], 1)
        return q
    qu, qv = augment(u, left), augment(v, right)
    core = torch.zeros(qu.shape[1], qv.shape[1], dtype=u.dtype, device=u.device)
    r = len(s)
    core[:r, :r] = torch.diag(s.to(u.dtype))
    core = core + (qu.mH @ left) @ (qv.mH @ right).mH
    uc, sc, vhc = svd(core, 'jacobi')
    keep = len(sc) if rank is None else rank
    if not 1 <= keep <= len(sc):
        raise ValueError('invalid update rank')
    return qu @ uc[:, :keep], sc[:keep], vhc[:keep] @ qv.mH


def append_columns(u, s, vh, columns, rank=None):
    """Append one or more columns using a standalone small-core decomposition."""
    from .solvers import svd
    if columns.ndim == 1:
        columns = columns[:, None]
    if columns.ndim != 2 or columns.shape[0] != u.shape[0]:
        raise ValueError('invalid appended columns')
    residual = _reorth(columns, u)
    ur, sr, _ = svd(residual, 'jacobi')
    cut = P._eps(u.dtype) * max(columns.shape) * columns.abs().max()
    count = min(int((sr > cut).sum()), u.shape[0] - u.shape[1])
    basis = torch.cat([u, ur[:, :count]], 1)
    old = torch.zeros(basis.shape[1], vh.shape[1], dtype=u.dtype, device=u.device)
    old[:len(s)] = s[:, None] * vh
    core = torch.cat([old, basis.mH @ columns], 1)
    uc, sc, vc = svd(core, 'jacobi')
    keep = len(sc) if rank is None else rank
    if not 1 <= keep <= len(sc):
        raise ValueError('invalid append rank')
    return basis @ uc[:, :keep], sc[:keep], vc[:keep]
