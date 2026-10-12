"""Explicit CPU diagnostics. Call outside measured stages; all arrays are inspectable."""
import numpy as np
import torch
from svdbench.metrics import validate


def host(x):
    if isinstance(x, torch.Tensor):
        if x.layout != torch.strided:
            raise ValueError('Inspect sparse values()/indices() explicitly; no silent densification')
        return x.detach().cpu().resolve_conj().resolve_neg().numpy().copy()
    return np.array(x, copy=True)


def accuracy(case, factors):
    u, s, vh = (host(x) for x in factors)
    shape = case.array.shape
    batch = shape[:-2]
    k = s.shape[-1] if s.ndim else 0
    if not 1 <= k <= min(shape[-2:]) or s.shape != (*batch, k):
        raise ValueError('Invalid singular-value shape')
    if u.shape != (*batch, shape[-2], k) or vh.shape != (*batch, k, shape[-1]):
        raise ValueError('Use reduced or truncated U, s, Vh with matching shapes')
    return validate(case.array, (u, s, vh), case.reference)


def spectrum_errors(case, singular_values):
    s = host(singular_values)
    ref = np.stack([r[1] for r in case.reference])
    if case.array.ndim == 2:
        ref = ref[0]
    ref = ref[..., :s.shape[-1]]
    floor = np.finfo(float).eps*max(case.array.shape[-2:])*np.maximum(ref[..., :1], np.finfo(float).tiny)
    absolute = abs(s-ref)
    relative = np.full_like(absolute, np.nan, dtype=float)
    np.divide(absolute, ref, out=relative, where=ref > floor)
    return {'reference_s': ref, 'absolute_error': absolute, 'relative_error': relative,
            'resolved': ref > floor}


def orthogonality(q):
    q = host(q)
    q = q.astype(np.complex128 if np.iscomplexobj(q) else np.float64)
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError('Expected a nonempty column basis')
    g = q.conj().T @ q-np.eye(q.shape[1])
    return {'orthogonality_fro_normalized': float(np.linalg.norm(g)/np.sqrt(q.shape[1])),
            'orthogonality_max_entry': float(abs(g).max())}


def basis_quality(case, q, side='left'):
    if case.array.ndim != 2 or side not in ('left', 'right'):
        raise ValueError('Use one matrix and side left/right')
    q = host(q)
    a = case.array if side == 'left' else case.array.conj().T
    if q.ndim != 2 or q.shape[0] != a.shape[0]:
        raise ValueError('Basis shape does not match the chosen side')
    result = orthogonality(q)
    result['projection_residual_relative_fro'] = float(np.linalg.norm(a-q@(q.conj().T@a))/(np.linalg.norm(a) or 1.))
    # A repeated-value boundary does not define a unique leading subspace.
    ur, sr, vhr = case.reference[0]
    k = q.shape[1]
    gap_ok = k < len(sr) and sr[k-1]-sr[k] > 100*np.finfo(float).eps*max(a.shape)*(sr[0] or 1.)
    result['leading_subspace_defined'] = bool(gap_ok)
    result['largest_principal_angle_radians'] = None
    if gap_ok:
        target = ur[:, :k] if side == 'left' else vhr[:k].conj().T
        u, s, _ = np.linalg.svd(q, full_matrices=False)
        if len(s) == k and s[-1] > np.finfo(float).eps*max(q.shape)*s[0]:
            cosines = np.linalg.svd(target.conj().T@u, compute_uv=False)
            result['largest_principal_angle_radians'] = float(np.arccos(np.clip(cosines.min(), 0, 1)))
    return result


def residual_arrays(case, factors):
    """Elementwise and per-triplet errors for a single matrix, in reference precision."""
    if case.array.ndim != 2:
        raise ValueError('Inspect one batch member at a time')
    u,s,vh = (host(x) for x in factors)
    dtype = np.complex128 if np.iscomplexobj(case.array) else np.float64
    a,u,vh = case.array.astype(dtype),u.astype(dtype),vh.astype(dtype)
    s = s.astype(np.float64)
    left,right = a@vh.conj().T-u*s, a.conj().T@u-vh.conj().T*s
    return dict(reconstruction_residual=a-(u*s)@vh, left_triplet_residual=left,
                right_triplet_residual=right, left_triplet_norms=np.linalg.norm(left,axis=0),
                right_triplet_norms=np.linalg.norm(right,axis=0),
                U_gram_error=u.conj().T@u-np.eye(len(s)), V_gram_error=vh@vh.conj().T-np.eye(len(s)))


def check_accuracy(metrics, **limits):
    """User-chosen metric limits; unavailable metrics fail rather than silently pass."""
    worst = metrics['worst']
    failures = []
    for key,limit in limits.items():
        if key not in worst: raise KeyError(f'Unknown accuracy metric: {key}')
        if not np.isfinite(limit) or limit<0: raise ValueError('Limits must be finite and nonnegative')
        value = worst[key]
        if value is None or not np.isfinite(value) or value>limit:
            failures.append(dict(metric=key, actual=value, limit=limit))
    if not worst.get('singular_values_nonnegative_sorted',False):
        failures.append(dict(metric='singular_values_nonnegative_sorted',actual=False,limit=True))
    return dict(passed=not failures, failures=failures, limits=limits)


def tensor_statistics(value):
    a = host(value)
    finite = np.isfinite(a)
    x = a[finite].astype(np.complex128 if np.iscomplexobj(a) else np.float64)
    result = dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes, elements=a.size,
                  finite=int(finite.sum()), nonfinite=int((~finite).sum()), zeros=int((a == 0).sum()))
    if x.size:
        z = abs(x)
        scale = z.max()
        result.update(abs_min=float(z.min()), abs_max=float(scale), abs_mean=float(z.mean()),
                      abs_std=float(z.std()), norm_fro=float(scale*np.linalg.norm(x/scale)) if scale else 0.,
                      mean_real=float(x.real.mean()), mean_imag=float(x.imag.mean()),
                      abs_quantiles=dict(zip(('p0','p25','p50','p75','p100'), np.quantile(z,[0,.25,.5,.75,1]).tolist())))
    return result
