"""Validation on the actual rounded input, outside all timed regions."""
import numpy as np
import scipy.linalg


def reference(a):
    dtype = np.complex128 if np.iscomplexobj(a) else np.float64
    arrays = a[None] if a.ndim == 2 else a
    return [scipy.linalg.svd(x.astype(dtype), full_matrices=False,
                            lapack_driver="gesvd", check_finite=True) for x in arrays]


def validate(a, result, refs):
    u, s, vh = result
    arrays = a[None] if a.ndim == 2 else a
    if a.ndim == 2:
        s = s[None]
        u, vh = (u[None], vh[None]) if u is not None else (None, None)
    rows = []
    for i, (x, (ur, sr, vhr)) in enumerate(zip(arrays, refs)):
        dtype = np.complex128 if np.iscomplexobj(x) else np.float64
        x = x.astype(dtype)
        k = s.shape[-1]
        si = s[i].astype(np.float64)
        if not np.isfinite(si).all() or (u is not None and
                (not np.isfinite(u[i]).all() or not np.isfinite(vh[i]).all())):
            raise ValueError("Non-finite SVD output")
        norm = np.linalg.norm(x)
        denom = norm if norm > 0 else 1.0
        top = sr[0] if sr[0] > 0 else 1.0
        floor = np.finfo(np.float64).eps * max(x.shape) * top
        resolved = sr[:k] > floor
        d = {
            "spectrum_error_over_sigma1": float(np.max(np.abs(si - sr[:k])) / top),
            "singular_values_nonnegative_sorted": bool(np.all(si >= 0) and np.all(np.diff(si) <= 0)),
            "resolved_spectrum_max_relative_error": float(np.max(np.abs(si[resolved] - sr[:k][resolved]) / sr[:k][resolved])) if resolved.any() else None,
            "reference_tail_relative_fro": float(np.linalg.norm(sr[k:]) / denom),
        }
        if u is not None:
            # Check complete bases for full mode; truncate only for reconstruction.
            ui, vi = u[i].astype(dtype), vh[i].astype(dtype)
            uk, vk = ui[:, :k], vi[:k, :]
            residual = np.linalg.norm(x - (uk * si) @ vk) / denom
            tail = d["reference_tail_relative_fro"]
            d.update({
                "reconstruction_relative_fro": float(residual),
                "orthogonality_u_fro_over_sqrt_cols": float(np.linalg.norm(ui.conj().T @ ui - np.eye(ui.shape[1])) / np.sqrt(ui.shape[1])),
                "orthogonality_v_fro_over_sqrt_rows": float(np.linalg.norm(vi @ vi.conj().T - np.eye(vi.shape[0])) / np.sqrt(vi.shape[0])),
                "left_triplet_relative_fro": float(np.linalg.norm(x @ vk.conj().T - uk * si) / denom),
                "right_triplet_relative_fro": float(np.linalg.norm(x.conj().T @ uk - vk.conj().T * si) / denom),
                "residual_over_optimal_tail": float(residual / tail) if tail > floor / denom else None,
            })
            # Compare subspaces, not signs/phases; a cut through a cluster is undefined.
            gap_ok = k < len(sr) and sr[k-1] - sr[k] > 100 * floor
            if gap_ok:
                vref, vc = vhr[:k].conj().T, vk.conj().T
                d["right_subspace_leakage_fro_over_sqrt_k"] = float(np.linalg.norm(vc - vref @ (vref.conj().T @ vc)) / np.sqrt(k))
            else:
                d["right_subspace_leakage_fro_over_sqrt_k"] = None
        rows.append(d)
    # Per-matrix diagnostics are retained; aggregate numeric fields are worst case.
    worst = {}
    for key in rows[0]:
        values = [r[key] for r in rows if r[key] is not None]
        worst[key] = (all(values) if isinstance(values[0], bool) else max(values)) if values else None
    return {"worst": worst, "per_matrix": rows}
