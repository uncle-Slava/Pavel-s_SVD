"""Template: your own SVD from scratch. Copy this file, change the body, check it:

    python -m svdlab check examples/my_svd.py --quick              # small matrices, CPU
    python -m svdlab check examples/my_svd.py --compare torch      # full set, next to torch
    python -m svdlab check examples/my_svd.py --device cuda        # on the GPU

The starter is a one-sided (Hestenes) Jacobi SVD in plain torch: rotate pairs of columns
of W = A V until all columns are orthogonal; then s = column norms, U = W / s.
All disjoint pairs of a round-robin round are rotated at once, so it also runs on the GPU.
"""
import torch


def svd(a, rank):
    """a: (m, n) torch tensor, real or complex; return U (m, k), s (k,), Vh (k, n) with k >= rank."""
    wide = a.shape[0] < a.shape[1]
    w = (a.mH if wide else a).clone().resolve_conj()   # work on a tall matrix
    m, n = w.shape
    v = torch.eye(n, dtype=w.dtype, device=w.device)
    real = w.real.dtype if w.is_complex() else w.dtype
    eps, tiny = torch.finfo(real).eps, torch.finfo(real).tiny
    tol = 10 * eps
    rounds = list(round_robin(n, w.device))

    for sweep in range(60):
        off = torch.zeros((), dtype=real, device=w.device)
        for p, q in rounds:
            wp, wq = w[:, p], w[:, q]
            alpha, beta = wp.abs().square().sum(0), wq.abs().square().sum(0)
            gamma = (wp.conj() * wq).sum(0)        # <w_p, w_q> for every pair of this round
            g = gamma.abs()
            correlation = g / (alpha * beta).sqrt().clamp_min(tiny)
            off = torch.maximum(off, correlation.max())
            zeta = (beta - alpha) / (2 * g.clamp_min(tiny))
            t = torch.where(zeta == 0, 1., torch.sign(zeta) / (zeta.abs() + torch.sqrt(1 + zeta ** 2)))
            c = torch.rsqrt(1 + t ** 2)
            s = torch.where(correlation > tol, c * t, 0.)
            c = torch.where(correlation > tol, c, 1.)
            phase = torch.where(g > 0, gamma / g.clamp_min(tiny), 1.)
            for x in (w, v):                      # same rotation on W and on V
                xp, xq = x[:, p], x[:, q]
                x[:, p] = c * xp - s * phase.conj() * xq
                x[:, q] = s * phase * xp + c * xq
        if off <= tol:                            # every pair orthogonal to working precision
            break

    sigma = torch.linalg.vector_norm(w, dim=0)
    order = torch.argsort(sigma, descending=True)
    sigma, w, v = sigma[order], w[:, order], v[:, order]
    u = w / torch.where(sigma > 0, sigma, 1.)
    u, sigma, vh = u[:, :rank], sigma[:rank], v[:, :rank].mH
    return (vh.mH, sigma, u.mH) if wide else (u, sigma, vh)


def round_robin(n, device):
    """Rounds of disjoint column pairs (p, q) covering every pair once per sweep."""
    players = list(range(n + n % 2))
    for _ in range(len(players) - 1):
        half = len(players) // 2
        pairs = [(min(x, y), max(x, y)) for x, y in zip(players[:half], reversed(players[half:])) if max(x, y) < n]
        yield (torch.tensor([p for p, _ in pairs], device=device), torch.tensor([q for _, q in pairs], device=device))
        players = [players[0], players[-1], *players[1:-1]]


if __name__ == '__main__':                        # python examples/my_svd.py: a one-matrix smoke test
    a = torch.randn(6, 4, dtype=torch.float64)
    u, s, vh = svd(a, 4)
    print('s =', s)
    print('|A - U S Vh| =', float(torch.linalg.norm(a - u * s @ vh)))
