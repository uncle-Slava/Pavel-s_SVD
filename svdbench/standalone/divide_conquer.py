"""Recursive bidiagonal SVD and Cuppen tridiagonal eigensolver.

Merges solve diagonal-plus-rank-one secular equations using pole-relative
bisection and a product-form eigenvector correction. This is transparent Python,
not LAPACK's optimized rational secular solver. Merge arithmetic uses FP64 on the
input device; difficult SVD merges are explicitly refined by our Jacobi routine.
"""
import torch
from . import primitives as P


def rank_one_eigh(d, z, stats):
    """Eigenpairs of diag(d) + z z^T, ascending; real, unbatched, FP64."""
    n = len(d)
    order = d.argsort()
    d, z = d[order].clone(), z[order].clone()
    rot = torch.eye(n, dtype=d.dtype, device=d.device)
    scale = torch.maximum(d.abs().max(), (z * z).sum()).clamp_min(torch.finfo(d.dtype).tiny)
    tol = 16 * torch.finfo(d.dtype).eps * scale
    # Rotate coincident poles before deleting negligible rank-one components.
    anchor = 0
    for j in range(1, n):
        if bool((d[j] - d[anchor]).abs() <= tol):
            r = torch.hypot(z[anchor], z[j])
            if bool(r > 0):
                c, s = z[anchor] / r, z[j] / r
                a, b = rot[:, anchor].clone(), rot[:, j].clone()
                rot[:, anchor], rot[:, j] = c * a + s * b, -s * a + c * b
                z[anchor], z[j] = r, 0
        else:
            anchor = j
    keep = z.abs() > 16 * torch.finfo(d.dtype).eps * scale.sqrt()
    idx = torch.where(keep)[0]
    vals, basis = d.clone(), rot.clone()
    stats['secular_merges'] = stats.get('secular_merges', 0) + 1
    stats['deflated_poles'] = stats.get('deflated_poles', 0) + n - len(idx)
    if len(idx):
        poles, zz = d[idx], z[idx]
        k = len(idx)
        gap = torch.cat([poles[1:] - poles[:-1], (zz * zz).sum()[None]])
        dd = poles[:, None] - poles[None, :]
        fm = 1 + ((zz * zz)[:, None] / (dd - gap[None, :] / 2)).sum(0)
        left = (fm > 0) | (torch.arange(k, device=d.device) == k - 1)
        origin_idx = torch.where(left, torch.arange(k, device=d.device), torch.arange(k, device=d.device) + 1)
        origin = poles[origin_idx]
        lo = torch.where(left, torch.zeros_like(gap), -gap / 2)
        hi = torch.where(left, torch.where(fm > 0, gap / 2, gap), torch.zeros_like(gap))
        delta = poles[:, None] - origin[None, :]
        for _ in range(100):
            mid = lo + (hi - lo) / 2
            f = 1 + ((zz * zz)[:, None] / (delta - mid[None, :])).sum(0)
            hi = torch.where(f > 0, mid, hi)
            lo = torch.where(f <= 0, mid, lo)
        off = lo + (hi - lo) / 2
        diff = delta - off[None, :]
        # Reconstruct the update from computed roots; logarithms avoid product underflow.
        pole_diff = poles[:, None] - poles[None, :]
        pole_diff.fill_diagonal_(1)
        logz = 0.5 * (diff.abs().log().sum(1) - pole_diff.abs().log().sum(1))
        zh = zz.sign() * logz.exp()
        vec = zh[:, None] / diff
        vscale = vec.abs().amax(0)
        vec = vec / vscale
        vec = vec / (vec * vec).sum(0).sqrt()
        if not bool(torch.isfinite(vec).all()):
            raise ArithmeticError('unresolved secular pole')
        basis[:, idx] = rot[:, idx] @ vec
        vals[idx] = origin + off
    inv = order.argsort()
    outorder = vals.argsort()
    return vals[outorder], basis[inv][:, outorder]


def bidiagonal_dc(B, stats, leaf_size=4):
    """B is upper bidiagonal n x (n or n+1). Returns full U, s, full V."""
    n, m = B.shape
    if n <= leaf_size:
        if m > n:
            v, s, u, _ = P.jacobi_svd(B.T[None], max_sweeps=80)
            v = P.complete_basis(v, m)
        else:
            u, s, v, _ = P.jacobi_svd(B[None], max_sweeps=80)
        return u[0], s[0], v[0]
    split = n // 2
    u1, s1, v1 = bidiagonal_dc(B[:split, :split + 1], stats, leaf_size)
    u2, s2, v2 = bidiagonal_dc(B[split + 1:, split + 1:], stats, leaf_size)
    ul = torch.eye(n, dtype=B.dtype, device=B.device)
    ul[:split, :split], ul[split + 1:, split + 1:] = u1, u2
    vr = torch.zeros(m, m, dtype=B.dtype, device=B.device)
    vr[:split + 1, :split + 1], vr[split + 1:, split + 1:] = v1, v2
    C = ul.T @ B @ vr
    poles = torch.zeros(m, dtype=B.dtype, device=B.device)
    poles[:split], poles[split + 1:split + 1 + len(s2)] = s1 * s1, s2 * s2
    try:
        _, v = rank_one_eigh(poles, C[split], stats)
        # Recover sigma by applying C, avoiding square-rooting a cancelled root.
        w = C @ v
        ss = (w * w).sum(0).sqrt()
        order = ss.argsort(descending=True)
        v, ss, w = v[:, order], ss[order][:n], w[:, order][:, :n]
        keep = ss > 16 * torch.finfo(B.dtype).eps * max(n, m) * ss[:1]
        u = w / torch.where(keep, ss, torch.ones_like(ss))
        u = P.complete_null_columns(u[None], keep[None])[0]
        ss = torch.where(keep, ss, torch.zeros_like(ss))
        err = (u.T @ u - torch.eye(n, device=B.device, dtype=B.dtype)).abs().max()
        verr = (v.T @ v - torch.eye(m, device=B.device, dtype=B.dtype)).abs().max()
        residual = ((u * ss) @ v[:, :n].T - C).abs().max()
        if not bool(torch.isfinite(err + verr + residual)) or bool(torch.maximum(err, verr) > 1e-10) or bool(residual > 1e-11 * C.abs().max()):
            raise ArithmeticError('secular merge needs orthogonality refinement')
    except (ArithmeticError, RuntimeError):
        # This is a disclosed standalone merge refinement, never a vendor fallback.
        stats['jacobi_merge_refinements'] = stats.get('jacobi_merge_refinements', 0) + 1
        if m > n:
            vv, ss, uu, _ = P.jacobi_svd(C.T[None], max_sweeps=80)
            v = P.complete_basis(vv, m)[0]
        else:
            uu, ss, vv, _ = P.jacobi_svd(C[None], max_sweeps=80)
            v = vv[0]
        u, ss = uu[0], ss[0]
    return ul @ u, ss, vr @ v


def tridiagonal_dc(d, e, stats, leaf_size=4):
    n = len(d)
    if n <= leaf_size:
        h = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
        w, v, _ = P.jacobi_eigh(h[None], max_sweeps=80)
        return w[0], v[0]
    mid = n // 2
    rho = e[mid - 1].abs()
    sign = torch.where(e[mid - 1] >= 0, 1., -1.)
    d1, d2 = d[:mid].clone(), d[mid:].clone()
    d1[-1] -= rho; d2[0] -= rho
    w1, q1 = tridiagonal_dc(d1, e[:mid - 1], stats, leaf_size)
    w2, q2 = tridiagonal_dc(d2, e[mid:], stats, leaf_size)
    dd = torch.cat([w1, w2])
    z = rho.sqrt() * torch.cat([q1[-1], sign * q2[0]])
    try:
        w, v = rank_one_eigh(dd, z, stats)
        err = (v.T @ v - torch.eye(n, dtype=d.dtype, device=d.device)).abs().max()
        if bool(err > 1e-10):
            raise ArithmeticError('secular orthogonality')
    except ArithmeticError:
        stats['jacobi_eigen_refinements'] = stats.get('jacobi_eigen_refinements', 0) + 1
        h = torch.diag(dd) + z[:, None] * z[None, :]
        w, v, _ = P.jacobi_eigh(h[None], max_sweeps=80)
        w, v = w[0], v[0]
    return w, torch.cat([q1 @ v[:mid], q2 @ v[mid:]], 0)
