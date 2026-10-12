"""
Batched linear-algebra primitives with NO torch.linalg calls.

Everything is vectorised over the batch dimension; loops run only over the
matrix dimension (n or m), the way a one-thread-block-per-matrix kernel
would loop.  Real and complex dtypes.  Shapes: A (b, m, n).

  householder_vec        reflector for a batch of vectors
  qr                     Householder QR -> Q (b,m,n), R (b,n,n)
  bidiag                 A = Q B Pᴴ, B real upper bidiagonal (d, e)
  tridiag                A = Q T Qᴴ, T real symmetric tridiagonal (d, e)
  cholesky               unblocked, returns L and a failure mask
  trsm_lower / trsm_upper  triangular solves (forward / backward substitution)
  bdsqr                  implicit-shift QR SVD of a batch of bidiagonals
                         (Golub–Kahan step, Wilkinson / zero shift, per-member
                         deflation via masks)
  jacobi_svd             one-sided Jacobi, round-robin ordering (batched)
  jacobi_eigh            Hermitian eig via two-sided cyclic Jacobi (syevj structure)
"""
# Adapted from the user-owned /root/svdbench/gpubench/custom/linalg.py
# retrieved 2026-10-11; numerical changes documented in docs/STANDALONE.md.
from __future__ import annotations
import torch


def _real_dtype(dt):
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}.get(dt, dt)


def _eps(dt):
    return torch.finfo(_real_dtype(dt)).eps


def _hypot(a, b):
    return torch.hypot(a, b)


# ---------------------------------------------------------------------------
# Householder
# ---------------------------------------------------------------------------
def householder_vec(x):
    """x: (b, k).  Returns v (b, k) with v[:,0]=1, tau (b,), beta (b,) real,
    such that (I − tau v vᴴ)ᴴ x = beta e₁   (LAPACK xLARFG convention)."""
    b, k = x.shape
    alpha = x[:, 0]
    xnorm = torch.sqrt((x[:, 1:].abs() ** 2).sum(-1)) if k > 1 else torch.zeros(b, dtype=_real_dtype(x.dtype), device=x.device)
    ar = alpha.real if x.is_complex() else alpha
    ai = alpha.imag if x.is_complex() else torch.zeros_like(ar)
    beta = -torch.sign(ar + (ar == 0)) * _hypot(alpha.abs(), xnorm)
    trivial = (xnorm == 0) & (ai == 0)
    beta = torch.where(trivial, ar, beta)
    denom = alpha - beta.to(x.dtype)
    denom = torch.where(trivial, torch.ones_like(denom), denom)
    tau = (beta.to(x.dtype) - alpha) / beta.to(x.dtype).where(beta != 0, torch.ones_like(beta.to(x.dtype)))
    tau = torch.where(trivial, torch.zeros_like(tau), tau)
    v = x / denom.unsqueeze(-1)
    v[:, 0] = 1.0
    v = torch.where(trivial.unsqueeze(-1), torch.zeros_like(v), v)
    return v, tau, beta


def _apply_left(A, v, tau):
    """A (b, k, c) <- (I − tau v vᴴ) A"""
    w = (v.conj().unsqueeze(-1) * A).sum(-2)                 # (b, c) = vᴴ A
    return A - tau.view(-1, 1, 1) * v.unsqueeze(-1) * w.unsqueeze(-2)


def _apply_right(A, v, tau):
    """A (b, r, k) <- A (I − tau v vᴴ)"""
    w = (A * v.unsqueeze(-2)).sum(-1)                        # (b, r) = A v
    return A - tau.view(-1, 1, 1) * w.unsqueeze(-1) * v.conj().unsqueeze(-2)


def qr(A):
    """Householder QR, batched.  Returns Q (b, m, n), R (b, n, n)."""
    A = A.clone()
    b, m, n = A.shape
    vs, taus = [], []
    for j in range(n):
        v, tau, beta = householder_vec(A[:, j:, j])
        A[:, j:, j + 1:] = _apply_left(A[:, j:, j + 1:], v, tau.conj())
        A[:, j, j] = beta.to(A.dtype); A[:, j + 1:, j] = 0
        vs.append(v); taus.append(tau)
    R = A[:, :n, :].triu()
    Q = torch.zeros(b, m, n, dtype=A.dtype, device=A.device)
    Q[:, :n, :] = torch.eye(n, dtype=A.dtype, device=A.device)
    for j in range(n - 1, -1, -1):                            # backward accumulation
        Q[:, j:, :] = _apply_left(Q[:, j:, :], vs[j], taus[j])
    return Q, R


def bidiag(A):
    """A (b, m, n), m >= n  ->  d (b, n), e (b, n-1) real, Q (b, m, n), P (b, n, n)
    with A = Q (diag(d) + superdiag(e)) Pᴴ."""
    A = A.clone()
    b, m, n = A.shape
    rd = _real_dtype(A.dtype)
    vs, taus = [], []
    P = torch.eye(n, dtype=A.dtype, device=A.device).expand(b, n, n).clone()
    d = torch.zeros(b, n, dtype=rd, device=A.device)
    e = torch.zeros(b, max(n - 1, 0), dtype=rd, device=A.device)
    for j in range(n):
        v, tau, beta = householder_vec(A[:, j:, j])
        A[:, j:, j + 1:] = _apply_left(A[:, j:, j + 1:], v, tau.conj())
        A[:, j, j] = beta.to(A.dtype); A[:, j + 1:, j] = 0
        vs.append(v); taus.append(tau)
        d[:, j] = beta
        if j + 1 < n:
            v, tau, beta = householder_vec(A[:, j, j + 1:].conj())
            A[:, j + 1:, j + 1:] = _apply_right(A[:, j + 1:, j + 1:], v, tau)
            A[:, j, j + 1] = beta.to(A.dtype); A[:, j, j + 2:] = 0
            P[:, :, j + 1:] = _apply_right(P[:, :, j + 1:], v, tau)
            e[:, j] = beta
    Q = torch.zeros(b, m, n, dtype=A.dtype, device=A.device)
    Q[:, :n, :] = torch.eye(n, dtype=A.dtype, device=A.device)
    for j in range(n - 1, -1, -1):
        Q[:, j:, :] = _apply_left(Q[:, j:, :], vs[j], taus[j])
    return d, e, Q, P


def tridiag(H):
    """Hermitian H (b, n, n) -> d (b, n), e (b, n-1) real, Q (b, n, n), H = Q T Qᴴ."""
    A = H.clone()
    b, n, _ = A.shape
    rd = _real_dtype(A.dtype)
    Q = torch.eye(n, dtype=A.dtype, device=A.device).expand(b, n, n).clone()
    e = torch.zeros(b, max(n - 1, 0), dtype=rd, device=A.device)
    for j in range(n - 1):
        v, tau, beta = householder_vec(A[:, j + 1:, j])
        blk = A[:, j + 1:, j + 1:]
        blk = _apply_left(blk, v, tau.conj())                 # Hᴴ A
        blk = _apply_right(blk, v, tau)                       # (Hᴴ A) H
        A[:, j + 1:, j + 1:] = blk
        A[:, j + 1:, j] = 0; A[:, j + 1, j] = beta.to(A.dtype)
        A[:, j, j + 1:] = 0; A[:, j, j + 1] = beta.to(A.dtype)
        Q[:, :, j + 1:] = _apply_right(Q[:, :, j + 1:], v, tau)
        e[:, j] = beta
    d = torch.diagonal(A, dim1=-2, dim2=-1).real if A.is_complex() else torch.diagonal(A, dim1=-2, dim2=-1)
    return d.clone(), e, Q


# ---------------------------------------------------------------------------
# Cholesky / triangular solves / CholeskyQR
# ---------------------------------------------------------------------------
def cholesky(G):
    """G (b, n, n) Hermitian PD -> L lower (G = L Lᴴ), fail (b,) bool."""
    b, n, _ = G.shape
    L = torch.zeros_like(G)
    fail = torch.zeros(b, dtype=torch.bool, device=G.device)
    for j in range(n):
        s = (L[:, j, :j].abs() ** 2).sum(-1)
        piv = (G[:, j, j].real if G.is_complex() else G[:, j, j]) - s
        fail |= piv <= 0
        ljj = torch.sqrt(piv.clamp_min(torch.finfo(piv.dtype).tiny))
        L[:, j, j] = ljj.to(G.dtype)
        if j + 1 < n:
            r = G[:, j + 1:, j] - (L[:, j + 1:, :j] * L[:, j, :j].conj().unsqueeze(-2)).sum(-1)
            L[:, j + 1:, j] = r / ljj.unsqueeze(-1).to(G.dtype)
    return L, fail


def trsm_lower(L, B):
    """Solve L X = B, L lower (b, n, n), B (b, n, c)."""
    n = L.shape[-1]
    X = torch.zeros_like(B)
    for i in range(n):
        acc = (L[:, i, :i].unsqueeze(-1) * X[:, :i, :]).sum(-2) if i > 0 else 0
        X[:, i, :] = (B[:, i, :] - acc) / L[:, i, i].unsqueeze(-1)
    return X


def trsm_upper(U, B):
    """Solve U X = B, U upper (b, n, n)."""
    n = U.shape[-1]
    X = torch.zeros_like(B)
    for i in range(n - 1, -1, -1):
        acc = (U[:, i, i + 1:].unsqueeze(-1) * X[:, i + 1:, :]).sum(-2) if i + 1 < n else 0
        X[:, i, :] = (B[:, i, :] - acc) / U[:, i, i].unsqueeze(-1)
    return X


# ---------------------------------------------------------------------------
# Givens helpers (batched scalars)
# ---------------------------------------------------------------------------
def _givens(a, b):
    """c, s, r with [a b][c s; -s c] = [r 0]; a, b real (batch,)."""
    r = _hypot(a, b)
    safe = r > 0
    c = torch.where(safe, a / r.where(safe, torch.ones_like(r)), torch.ones_like(r))
    s = torch.where(safe, b / r.where(safe, torch.ones_like(r)), torch.zeros_like(r))
    return c, s, r


def _rot_cols(M, i, j, c, s):
    """M (b, ., n): columns i, j <- c*Mi + s*Mj, -s*Mi + c*Mj  (c, s (b,) real)."""
    c = c.view(-1, 1).to(M.dtype); s = s.view(-1, 1).to(M.dtype)
    Mi, Mj = M[:, :, i].clone(), M[:, :, j].clone()
    M[:, :, i] = c * Mi + s * Mj
    M[:, :, j] = -s * Mi + c * Mj


# ---------------------------------------------------------------------------
# bdsqr: batched implicit-shift QR SVD of upper bidiagonals
# ---------------------------------------------------------------------------
def _active_window(e):
    """Per member: hi = index of the last nonzero e (+1 = bottom row of the
    active block), lo = top of the unreduced block ending at hi.
    Members with no nonzero e get hi = 0 (inactive)."""
    b, n1 = e.shape
    nz = e != 0
    idx = torch.arange(n1, device=e.device).expand(b, n1)
    any_nz = nz.any(-1)
    hi_e = torch.where(nz, idx, torch.full_like(idx, -1)).max(-1).values          # last nonzero index
    zero_before = (~nz) & (idx < hi_e.unsqueeze(-1))
    last_zero = torch.where(zero_before, idx, torch.full_like(idx, -1)).max(-1).values
    lo = last_zero + 1
    hi = hi_e + 1
    lo = torch.where(any_nz, lo, torch.zeros_like(lo)); hi = torch.where(any_nz, hi, torch.zeros_like(hi))
    return lo, hi, any_nz


def _gather(x, i):
    return x.gather(1, i.clamp(0, x.shape[1] - 1).unsqueeze(-1)).squeeze(-1)


def bdsqr(d, e, compute_uv=True, max_iter=None, tol=None):
    """d (b, n), e (b, n-1) real.  Returns s (b, n) descending, U (b, n, n),
    V (b, n, n) with B = U diag(s) Vᵀ.  Golub–Kahan implicit-shift sweeps,
    all members advanced in lock-step with per-member [lo, hi] windows."""
    d = d.clone(); e = e.clone()
    b, n = d.shape
    eps = torch.finfo(d.dtype).eps
    tol = tol or 10 * eps
    max_iter = max_iter if max_iter is not None else 60 * n
    if n == 1:
        v = torch.sign(d + (d == 0)).unsqueeze(-1)
        bdsqr.last_iters = 0
        return d.abs(), torch.ones_like(v), v
    U = torch.eye(n, dtype=d.dtype, device=d.device).expand(b, n, n).clone() if compute_uv else None
    V = torch.eye(n, dtype=d.dtype, device=d.device).expand(b, n, n).clone() if compute_uv else None
    it = 0
    for it in range(1, max_iter + 1):
        # deflation
        small = e.abs() <= tol * (d[:, :-1].abs() + d[:, 1:].abs())
        e = torch.where(small, torch.zeros_like(e), e)
        lo, hi, active = _active_window(e)
        if not bool(active.any()):
            break
        # Wilkinson shift from trailing 2x2 of BᵀB on [lo, hi]
        dm, em, dn = _gather(d, hi - 1), _gather(e, hi - 1), _gather(d, hi)
        em1 = torch.where(hi - 2 >= lo, _gather(e, hi - 2), torch.zeros_like(em))
        t11 = dm * dm + em1 * em1; t12 = dm * em; t22 = dn * dn + em * em
        delta = 0.5 * (t11 - t22)
        sgn = torch.where(delta >= 0, torch.ones_like(delta), -torch.ones_like(delta))
        mu = t22 - t12 * t12 / (delta + sgn * _hypot(delta, t12) + torch.finfo(d.dtype).tiny)
        # zero-shift (Demmel–Kahan) when the block is badly scaled
        blk = (torch.arange(n, device=d.device).expand(b, n) >= lo.unsqueeze(-1)) & (torch.arange(n, device=d.device).expand(b, n) <= hi.unsqueeze(-1))
        smin = torch.where(blk, d.abs(), torch.full_like(d, float("inf"))).min(-1).values
        smax = torch.where(blk, d.abs(), torch.zeros_like(d)).max(-1).values
        mu = torch.where((mu <= 0) | (n * tol * (smin / smax.clamp_min(torch.finfo(d.dtype).tiny)) <= max(eps, 0.01 * tol)), torch.zeros_like(mu), mu)
        # chase, k = 0..n-2, masked to [lo, hi-1]
        y = torch.zeros_like(mu); z = torch.zeros_like(mu); bulge = torch.zeros_like(mu)
        for k in range(n - 1):
            act = active & (k >= lo) & (k < hi)
            start = act & (k == lo)
            y = torch.where(start, torch.where(mu == 0, d[:, k], d[:, k] * d[:, k] - mu), y)
            z = torch.where(start, torch.where(mu == 0, e[:, k], d[:, k] * e[:, k]), z)
            # right rotation on columns (k, k+1)
            c, s, _ = _givens(y, z)
            c = torch.where(act, c, torch.ones_like(c)); s = torch.where(act, s, torch.zeros_like(s))
            if k > 0:
                e[:, k - 1] = c * e[:, k - 1] + s * bulge
            dk, ek, dk1 = d[:, k].clone(), e[:, k].clone(), d[:, k + 1].clone()
            d[:, k] = c * dk + s * ek
            e[:, k] = -s * dk + c * ek
            fill = s * dk1
            d[:, k + 1] = c * dk1
            if compute_uv:
                _rot_cols(V, k, k + 1, c, s)
            # left rotation on rows (k, k+1)
            c, s, r = _givens(d[:, k], fill)
            c = torch.where(act, c, torch.ones_like(c)); s = torch.where(act, s, torch.zeros_like(s))
            d[:, k] = torch.where(act, r, d[:, k])
            ek, dk1 = e[:, k].clone(), d[:, k + 1].clone()
            e[:, k] = c * ek + s * dk1
            d[:, k + 1] = -s * ek + c * dk1
            if compute_uv:
                _rot_cols(U, k, k + 1, c, s)
            if k + 1 < n - 1:
                bulge = s * e[:, k + 1]
                e[:, k + 1] = c * e[:, k + 1]
                y = e[:, k]; z = bulge
    if bool((e.abs() > tol * (d[:, :-1].abs() + d[:, 1:].abs())).any()):
        raise RuntimeError(f"bidiagonal QR did not converge in {max_iter} iterations")
    s_vals = d.abs()
    if compute_uv:
        V = V * torch.sign(d + (d == 0)).unsqueeze(-2).to(V.dtype)
    order = torch.argsort(-s_vals, dim=-1)
    s_vals = s_vals.gather(1, order)
    if compute_uv:
        U = U.gather(2, order.unsqueeze(1).expand_as(U)); V = V.gather(2, order.unsqueeze(1).expand_as(V))
    bdsqr.last_iters = it
    return s_vals, U, V


# ---------------------------------------------------------------------------
# Jacobi
# ---------------------------------------------------------------------------
def _round_robin(n):
    players = list(range(n))
    for _ in range(n - 1):
        yield [(min(players[i], players[n - 1 - i]), max(players[i], players[n - 1 - i])) for i in range(n // 2)]
        players = [players[0]] + [players[-1]] + players[1:-1]


def jacobi_svd(A, tol=None, max_sweeps=30):
    """One-sided Jacobi, round-robin parallel ordering, batched.
    Returns U (b,m,n), s (b,n) descending, V (b,n,n), sweeps."""
    b, m, n = A.shape
    padded = n % 2 == 1
    W = A.clone()
    if padded:
        W = torch.cat([W, torch.zeros(b, m, 1, dtype=A.dtype, device=A.device)], -1); n += 1
    V = torch.eye(n, dtype=W.dtype, device=W.device).expand(b, n, n).clone()
    tol = tol or 10 * _eps(A.dtype)
    sched = [(torch.tensor([p for p, _ in prs], device=A.device), torch.tensor([q for _, q in prs], device=A.device)) for prs in _round_robin(n)]
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        off = torch.zeros((), dtype=_real_dtype(A.dtype), device=A.device)
        for P, Q in sched:
            wp, wq = W[:, :, P], W[:, :, Q]
            app = (wp.abs() ** 2).sum(1); aqq = (wq.abs() ** 2).sum(1)
            apq = (wp.conj() * wq).sum(1)
            den = app.sqrt() * aqq.sqrt()
            ratio = apq.abs() / den.clamp_min(torch.finfo(den.dtype).tiny)
            # skip pairs with a (numerically) null column: nothing to rotate
            floor = _eps(A.dtype) ** 2 * torch.maximum(app, aqq).amax(-1, keepdim=True)
            active = (ratio > tol) & (torch.minimum(app, aqq) > floor)
            off = torch.maximum(off, torch.where(active, ratio, torch.zeros_like(ratio)).max())
            a = torch.where(active, apq.abs(), torch.ones_like(den))
            tau = (aqq - app) / (2 * a)
            t = torch.sign(tau) / (tau.abs() + torch.hypot(tau, torch.ones_like(tau)))
            t = torch.where(tau == 0, torch.ones_like(t), t)
            c = 1 / (1 + t * t).sqrt()
            s = (t * c).to(A.dtype) * (apq / a.to(A.dtype))
            c = torch.where(active, c, torch.ones_like(c)).to(A.dtype)
            s = torch.where(active, s, torch.zeros_like(s))
            c = c.unsqueeze(1); s = s.unsqueeze(1)
            W[:, :, P], W[:, :, Q] = c * wp - s.conj() * wq, s * wp + c * wq
            vp, vq = V[:, :, P], V[:, :, Q]
            V[:, :, P], V[:, :, Q] = c * vp - s.conj() * vq, s * vp + c * vq
        if bool(off <= tol):
            break
    if bool(off > tol):
        raise RuntimeError(f"one-sided Jacobi did not converge in {max_sweeps} sweeps")
    s_vals = torch.sqrt((W.abs() ** 2).sum(1))
    if padded:
        W, V, s_vals = W[:, :, :-1], V[:, :-1, :-1], s_vals[:, :-1]
    order = torch.argsort(-s_vals, dim=1)
    s_vals = s_vals.gather(1, order)
    W = W.gather(2, order.unsqueeze(1).expand_as(W)); V = V.gather(2, order.unsqueeze(1).expand_as(V))
    cut = _eps(A.dtype) * max(m, n) * s_vals[:, :1]
    keep = s_vals > cut
    U = W / torch.where(keep, s_vals, torch.ones_like(s_vals)).unsqueeze(1)
    U = complete_null_columns(U, keep)
    s_vals = torch.where(keep, s_vals, torch.zeros_like(s_vals))
    return U, s_vals, V, sweeps


def jacobi_eigh(H, tol=None, max_sweeps=30):
    """Hermitian H (b,n,n) -> w (b,n) ascending, V (b,n,n), sweeps.
    Two-sided cyclic Hermitian Jacobi (the syevj / syevjBatched structure):
    G <- Jᴴ G J with one plane rotation per pair, round-robin parallel ordering,
    V accumulated.  Convergence: every |g_pq| <= tol * max(|g_pp|, |g_qq|)
    (cuSOLVER's off(A) / ||A|| criterion, pairwise) - *absolute* in the size of
    the diagonal, so eigenvalues come out with error ~ eps * lambda_max, which is
    what the Gram route can deliver.  (Until 0.1.2 this was a one-sided Jacobi
    SVD of H itself: for a Gram matrix that squares the conditioning once more
    and never meets its relative stopping test in fp32 - sigma_min errors of
    600 % at cond 1e3.)"""
    b, n, _ = H.shape
    padded = n % 2 == 1
    G = H.clone()
    if padded:                                   # pad with a zero row/column: its pairs are never active
        Z = torch.zeros(b, n + 1, n + 1, dtype=H.dtype, device=H.device); Z[:, :n, :n] = G; G = Z; n += 1
    V = torch.eye(n, dtype=G.dtype, device=G.device).expand(b, n, n).clone()
    tol = tol or 10 * _eps(H.dtype)
    bidx = torch.arange(b, device=H.device)[:, None]
    sched = [(torch.tensor([p for p, _ in prs], device=H.device), torch.tensor([q for _, q in prs], device=H.device)) for prs in _round_robin(n)]
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        off = torch.zeros((), dtype=_real_dtype(H.dtype), device=H.device)
        for P, Q in sched:
            gpp = G[bidx, P, P]; gqq = G[bidx, Q, Q]; gpq = G[bidx, P, Q]            # (b, k)
            gpp = gpp.real if gpp.is_complex() else gpp; gqq = gqq.real if gqq.is_complex() else gqq
            a = gpq.abs()
            scale = G.abs().amax((-2, -1))[:, None].expand_as(gpp)
            active = a > tol * scale
            off = torch.maximum(off, (a / scale.clamp_min(torch.finfo(scale.dtype).tiny)).max())
            a_safe = torch.where(active, a, torch.ones_like(a))
            tau = (gqq - gpp) / (2 * a_safe)
            t = torch.sign(tau) / (tau.abs() + torch.hypot(tau, torch.ones_like(tau)))
            t = torch.where(tau == 0, torch.ones_like(t), t)
            c = 1 / (1 + t * t).sqrt()
            s = (t * c).to(G.dtype) * (gpq / a_safe.to(G.dtype))                    # phase of g_pq
            c = torch.where(active, c, torch.ones_like(c)).to(G.dtype)
            s = torch.where(active, s, torch.zeros_like(s))
            # G <- G J   (columns)        J[p,p]=J[q,q]=c, J[q,p]=-conj(s), J[p,q]=s
            cc, ss = c.unsqueeze(1), s.unsqueeze(1)
            gp, gq = G[:, :, P], G[:, :, Q]
            G[:, :, P], G[:, :, Q] = cc * gp - ss.conj() * gq, ss * gp + cc * gq
            vp, vq = V[:, :, P], V[:, :, Q]
            V[:, :, P], V[:, :, Q] = cc * vp - ss.conj() * vq, ss * vp + cc * vq
            # G <- Jᴴ G   (rows)
            cc, ss = c.unsqueeze(2), s.unsqueeze(2)
            rp, rq = G[:, P, :], G[:, Q, :]
            G[:, P, :], G[:, Q, :] = cc * rp - ss * rq, ss.conj() * rp + cc * rq
        if bool(off <= tol):
            break
    if bool(off > tol):
        raise RuntimeError(f"Hermitian Jacobi did not converge in {max_sweeps} sweeps")
    w = G.diagonal(dim1=-2, dim2=-1)
    w = w.real if w.is_complex() else w
    if padded:
        w, V = w[:, :-1], V[:, :-1, :-1]
    order = torch.argsort(w, dim=-1)
    return w.gather(1, order), V.gather(2, order.unsqueeze(1).expand_as(V)), sweeps


def complete_null_columns(u, keep):
    """Complete zero columns using projected coordinate vectors; preserve live columns."""
    if bool(keep.all()):
        return u
    u = u.clone()
    b, m, n = u.shape
    u = torch.where(keep[:, None, :], u, torch.zeros_like(u))
    batch = torch.arange(b, device=u.device)
    for j in range(n):
        missing = ~keep[:, j]
        if not bool(missing.any()):
            continue
        # Choose the coordinate with largest remaining leverage. Two projections
        # reduce cancellation, including after many completed null columns.
        leverage = 1 - (u.abs() ** 2).sum(-1)
        idx = leverage.argmax(-1)
        v = torch.zeros(b, m, 1, dtype=u.dtype, device=u.device)
        v[batch, idx, 0] = 1
        for _ in range(2):
            v = v - u @ (u.mH @ v)
        length = (v.abs() ** 2).sum(-2, keepdim=True).sqrt()
        if bool((length[missing] < 10 * _eps(u.dtype)).any()):
            raise RuntimeError("null-space completion failed: supplied live columns are not independent")
        v = v / length.clamp_min(torch.finfo(length.dtype).tiny)
        u[:, :, j] = torch.where(missing[:, None], v[:, :, 0], u[:, :, j])
    return u


def complete_basis(q, size):
    """Extend an orthonormal basis to size columns (without a factorization call)."""
    if size == q.shape[-1]:
        return q
    b, m, k = q.shape
    if not k <= size <= m:
        raise ValueError("invalid basis size")
    z = torch.zeros(b, m, size - k, dtype=q.dtype, device=q.device)
    keep = torch.arange(size, device=q.device)[None, :].expand(b, size) < k
    return complete_null_columns(torch.cat([q, z], -1), keep)
