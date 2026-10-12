"""EDIT THIS FILE: visible numerical steps, swappable primitives, no hidden tracing.

The default QR and projected SVD are the existing standalone implementations.
Pass primitives='library' for optimized Torch controls. Inspection can solve a
projected SVD at every iteration; NullTrace omits those diagnostic-only solves.
"""
import torch
from svdbench.laboratory import NullTrace, basis_quality
from svdbench.standalone import primitives as P
from svdbench.standalone import svd


def orthogonalize(y, primitives='standalone'):
    if primitives == 'standalone':
        return P.qr(y[None])[0][0]
    if primitives == 'library':
        return torch.linalg.qr(y, mode='reduced')[0]
    raise ValueError('Unknown primitive implementation')


def small_svd(b, primitives='standalone'):
    if primitives == 'standalone': return svd(b, 'jacobi')
    if primitives == 'library': return torch.linalg.svd(b, full_matrices=False)
    raise ValueError('Unknown primitive implementation')


def product(a, x):
    return a@x if a.layout == torch.strided else torch.sparse.mm(a, x)


def randomized(a, trace=None, rank=8, oversample=8, iterations=2, seed=7,
               primitives='standalone', orthogonalizer=None, core_solver=None):
    """Single real/complex matrix; dense or COO. Q0 plus each power iterate is visible."""
    trace = trace if trace is not None else NullTrace()
    if a.ndim != 2 or not 1 <= rank <= min(a.shape) or oversample<0 or iterations<0:
        raise ValueError('Invalid randomized SVD parameters')
    orth = orthogonalizer or (lambda x: orthogonalize(x, primitives))
    core_svd = core_solver or (lambda x: small_svd(x, primitives))
    ell = min(rank+oversample, min(a.shape))
    ah = a.mH if a.layout == torch.strided else a.mH.coalesce()
    with trace.stage('random_sketch'):
        gen = torch.Generator(device=a.device).manual_seed(seed)
        omega = torch.randn(a.shape[1], ell, dtype=a.dtype, device=a.device, generator=gen)
    with trace.stage('A_times_sketch', iteration=0):
        y = product(a, omega)
    with trace.stage('orthogonalize_left', iteration=0):
        q = orth(y)

    def inspect(iteration):
        if trace.wants(iteration):
            # Diagnostic work is deliberately OUTSIDE measured algorithm stages.
            core = product(ah, q).mH
            uc, s, vh = core_svd(core)
            trace.checkpoint('iterate', iteration, tensors={'Q':q, 'projected':core},
                             scalars=basis_quality(trace.case, q),
                             factors=(q@uc[:,:rank],s[:rank],vh[:rank]))
    inspect(0)
    for iteration in range(1, iterations+1):
        # Replace either product or either orthogonalization independently.
        with trace.stage('adjoint_times_basis', iteration):
            z = product(ah, q)
        trace.checkpoint('before_right_qr', iteration, tensors={'Z':z})
        with trace.stage('orthogonalize_right', iteration):
            z = orth(z)
        with trace.stage('A_times_basis', iteration):
            y = product(a, z)
        trace.checkpoint('before_left_qr', iteration, tensors={'Y':y})
        with trace.stage('orthogonalize_left', iteration):
            q = orth(y)
        inspect(iteration)
    with trace.stage('project'):
        core = product(ah, q).mH
    with trace.stage('small_svd'):
        uc, s, vh = core_svd(core)
    with trace.stage('lift_vectors'):
        result = q@uc[:,:rank], s[:rank], vh[:rank]
    trace.checkpoint('final', factors=result)
    return result


def jacobi_rotation(w, v, p, q, tol):
    """One independent group of plane rotations. Edit/benchmark this function alone."""
    wp, wq = w[:,p], w[:,q]
    app, aqq = wp.abs().square().sum(0), wq.abs().square().sum(0)
    apq = (wp.conj()*wq).sum(0)
    den = app.sqrt()*aqq.sqrt()
    eps = torch.finfo(w.real.dtype).eps
    ratio = apq.abs()/den.clamp_min(torch.finfo(den.dtype).tiny)
    active = (ratio>tol) & (torch.minimum(app,aqq)>eps**2*torch.maximum(app,aqq).max())
    off = torch.where(active,ratio,0).max()
    magnitude = torch.where(active,apq.abs(),torch.ones_like(den))
    tau = (aqq-app)/(2*magnitude)
    t = torch.sign(tau)/(tau.abs()+torch.hypot(tau,torch.ones_like(tau)))
    t = torch.where(tau==0,torch.ones_like(t),t)
    c = torch.rsqrt(1+t*t)
    s = (t*c).to(w.dtype)*(apq/magnitude.to(w.dtype))
    c = torch.where(active,c,torch.ones_like(c)).to(w.dtype)
    s = torch.where(active,s,torch.zeros_like(s))
    w[:,p], w[:,q] = c*wp-s.conj()*wq, s*wp+c*wq
    vp, vq = v[:,p], v[:,q]
    v[:,p], v[:,q] = c*vp-s.conj()*vq, s*vp+c*vq
    return off


def jacobi(a, trace=None, max_sweeps=40, tol=None, record_rotations=False, require_convergence=True, **unused):
    """Standalone one-sided Jacobi with checkpoints per sweep and optional round."""
    trace = trace if trace is not None else NullTrace()
    if a.ndim != 2 or a.layout != torch.strided or max_sweeps<1:
        raise ValueError('Jacobi expects one dense matrix and positive max_sweeps')
    wide = a.shape[0]<a.shape[1]
    x = a.mH if wide else a
    scale = x.abs().max()
    scale = torch.where(scale>0,scale,torch.ones_like(scale))
    w = (x/scale).clone()
    m,n = w.shape
    v = torch.eye(n,dtype=a.dtype,device=a.device)
    eps = torch.finfo(a.real.dtype).eps
    tol = 10*eps if tol is None else tol
    if tol<=0: raise ValueError('Tolerance must be positive')
    schedule=[]
    for pairs in P._round_robin(n+n%2):
        pairs=[(p,q) for p,q in pairs if p<n and q<n]
        if pairs:
            schedule.append((torch.tensor([p for p,q in pairs],device=a.device),
                             torch.tensor([q for p,q in pairs],device=a.device)))

    def factors():
        s = w.abs().square().sum(0).sqrt()
        order = s.argsort(descending=True)
        s, ww, vv = s[order], w[:,order], v[:,order]
        live = s>eps*max(m,n)*s[:1]
        u = ww/torch.where(live,s,torch.ones_like(s))
        u = P.complete_null_columns(u[None],live[None])[0]
        s = torch.where(live,s,torch.zeros_like(s))*scale
        return (vv,s,u.mH) if wide else (u,s,vv.mH)

    converged=False
    for sweep in range(1,max_sweeps+1):
        off = torch.zeros((),dtype=a.real.dtype,device=a.device)
        for index,(p,q) in enumerate(schedule):
            with trace.stage('jacobi_rotation', sweep, round=index):
                off = torch.maximum(off,jacobi_rotation(w,v,p,q,tol))
            if record_rotations:
                trace.checkpoint(f'rotation_{index}',sweep,tensors={'W':w,'V':v,'p':p,'q':q})
        if trace.wants(sweep):
            trace.checkpoint('sweep',sweep,tensors={'W':w,'V':v},
                             scalars={'max_active_pair_correlation':off,'tolerance':tol},factors=factors())
        if bool(off<=tol):
            converged=True
            break
    trace.record('convergence',converged=converged,sweeps=sweep,tolerance=tol)
    if not converged and require_convergence:
        raise RuntimeError(f'Jacobi did not converge in {max_sweeps} sweeps; inspect the saved trace')
    with trace.stage('sort_and_normalize'):
        result=factors()
    trace.checkpoint('final',factors=result)
    return result
