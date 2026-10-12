"""Tucker/HOSVD, sequentially truncated HOSVD, and tensor-train SVD."""
import torch
from .solvers import svd


def unfold(x, mode):
    return x.movedim(mode, 0).reshape(x.shape[mode], -1)


def mode_product(x, matrix, mode):
    shape = list(x.shape)
    shape[mode] = matrix.shape[0]
    out = matrix @ unfold(x, mode)
    return out.reshape(shape[mode], *(x.shape[j] for j in range(x.ndim) if j != mode)).movedim(0, mode)


def tucker(x, ranks, sequential=False, algorithm='jacobi', **opts):
    if len(ranks) != x.ndim or any(not 1 <= r <= x.shape[i] for i, r in enumerate(ranks)):
        raise ValueError('one valid rank per tensor mode required')
    core, factors, info = x.clone(), [], []
    for mode, rank in enumerate(ranks):
        unfolding = unfold(core if sequential else x, mode)
        # For uncompressed tall unfoldings, complete left null space if requested.
        if rank > min(unfolding.shape):
            u, _, _ = svd(unfolding, algorithm, full_matrices=True, **opts)
        else:
            u, _, _ = svd(unfolding, algorithm, rank=rank, **opts)
        factors.append(u[:, :rank])
        info.append({'mode': mode, 'unfolding_shape': list(unfolding.shape),
                     'unfolding_bytes': unfolding.numel() * unfolding.element_size(),
                     'shares_source_storage': unfolding.untyped_storage().data_ptr() == (core if sequential else x).untyped_storage().data_ptr()})
        if sequential:
            core = mode_product(core, u[:, :rank].mH, mode)
    if not sequential:
        for mode, factor in enumerate(factors):
            core = mode_product(core, factor.mH, mode)
    return core, factors, info


def tucker_reconstruct(core, factors):
    x = core
    for mode, factor in enumerate(factors):
        x = mode_product(x, factor, mode)
    return x


def tensor_train(x, ranks, algorithm='jacobi', **opts):
    if len(ranks) != x.ndim - 1 or any(r < 1 for r in ranks):
        raise ValueError('positive rank for every internal TT bond required')
    cores, rest, prev, info = [], x, 1, []
    for j, requested in enumerate(ranks):
        mat = rest.reshape(prev * x.shape[j], -1)
        rank = min(requested, min(mat.shape))
        u, s, vh = svd(mat, algorithm, rank=rank, **opts)
        cores.append(u.reshape(prev, x.shape[j], rank))
        info.append({'unfolding_shape': list(mat.shape), 'rank': rank,
                     'unfolding_bytes': mat.numel() * mat.element_size()})
        rest = s[:, None] * vh
        prev = rank
    cores.append(rest.reshape(prev, x.shape[-1], 1))
    return cores, info


def tensor_train_reconstruct(cores):
    x = cores[0][0]
    for c in cores[1:]:
        x = torch.tensordot(x, c, dims=([-1], [0]))
    return x[..., 0]
