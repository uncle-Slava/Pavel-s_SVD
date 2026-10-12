"""Reproducible matrices and lazy independent references; no timed work here."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from svdbench.matrices import make_matrix
from svdbench.metrics import reference


class MatrixCase:
    def __init__(self, array, spec=None):
        self.array = np.array(array, copy=True, order='C')
        if self.array.ndim not in (2, 3) or min(self.array.shape) < 1:
            raise ValueError('Expected a nonempty matrix or batch of matrices')
        if self.array.dtype.name not in ('float32', 'float64', 'complex64', 'complex128'):
            raise ValueError('Use float32/64 or complex64/128 input')
        if not np.isfinite(self.array).all():
            raise ValueError('Input must be finite')
        self.array.flags.writeable = False
        self.spec = dict(spec or {})
        self.sha256 = hashlib.sha256(self.array.tobytes()).hexdigest()
        self._reference = None

    @classmethod
    def create(cls, m=96, n=64, family='low_rank', dtype='float64', seed=7, **parameters):
        spec = dict(m=m, n=n, family=family, dtype=dtype, seed=seed, **parameters)
        if min(m, n) < 1 or parameters.get('batch', 1) < 1:
            raise ValueError('Dimensions and batch must be positive')
        if family == 'low_rank' and not 0 <= parameters.get('rank', max(1, min(m, n)//8)) <= min(m, n):
            raise ValueError('Generating rank is outside the matrix dimensions')
        return cls(make_matrix(spec), spec)

    @classmethod
    def load(cls, path):
        return cls(np.load(path, allow_pickle=False), {'loaded_from': str(path)})

    @classmethod
    def from_spectrum(cls, singular_values, m=None, n=None, seed=7, dtype='float64'):
        s = np.asarray(singular_values, dtype=np.float64)
        if s.ndim != 1 or len(s) == 0 or not np.isfinite(s).all() or (s < 0).any():
            raise ValueError('Provide a finite, nonnegative singular-value vector')
        m, n = m or len(s), n or len(s)
        if len(s) > min(m, n):
            raise ValueError('Too many singular values for this shape')
        rng = np.random.default_rng(seed)
        def basis(rows):
            x = rng.standard_normal((rows, len(s)))
            if np.issubdtype(np.dtype(dtype), np.complexfloating):
                x = x + 1j*rng.standard_normal(x.shape)
            return np.linalg.qr(x, mode='reduced')[0]
        a = (basis(m)*s) @ basis(n).conj().T
        return cls(a.astype(dtype), dict(family='custom_spectrum', singular_values=s.tolist(),
                                        m=m, n=n, seed=seed, dtype=dtype))

    @property
    def reference(self):
        if self._reference is None:
            self._reference = reference(self.array)
        return self._reference

    def tensor(self, device='cpu', sparse=False):
        a = torch.from_numpy(self.array.copy()).to(device)
        if sparse:
            if a.ndim != 2:
                raise ValueError('Sparse conversion currently expects one matrix')
            return a.to_sparse().coalesce()
        return a

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path/'input.npy', self.array, allow_pickle=False)
        np.savez(path/'reference.npz', u=np.stack([r[0] for r in self.reference]),
                 s=np.stack([r[1] for r in self.reference]), vh=np.stack([r[2] for r in self.reference]))
