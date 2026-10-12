"""CPU-generated inputs; save the corpus to share identical bytes across hosts."""
import hashlib
from pathlib import Path

import numpy as np


def make_matrix(case):
    m, n = case["m"], case["n"]
    b = case.get("batch", 1)
    rng = np.random.default_rng(case.get("seed", 20261011))
    dtype = np.dtype(case.get("dtype", "float32"))
    if dtype not in [np.dtype(x) for x in ("float32", "float64", "complex64", "complex128")]:
        raise ValueError("Use float32/64 or complex64/128; mixed precision needs an explicit algorithm")

    def random(shape):
        a = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = (a + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        return a

    matrices = []
    family = case.get("family", "gaussian")
    r = min(m, n)
    for _ in range(b):
        if family == "gaussian":
            a = random((m, n)) / np.sqrt(max(m, n))
        elif family == "sparse":
            density = case.get("density", 0.05)
            if not 0 <= density <= 1:
                raise ValueError("density must lie in [0,1]")
            a = random((m, n)) / np.sqrt(max(m, n))
            a *= rng.random((m, n)) < density
        elif family == "zero":
            a = np.zeros((m, n))
        else:
            q, _ = np.linalg.qr(random((m, r)), mode="reduced")
            v, _ = np.linalg.qr(random((n, r)), mode="reduced")
            if family == "geometric":
                s = np.geomspace(1, 1 / case.get("condition", 1e6), r)
            elif family == "clustered":
                cluster = case.get("cluster_size", 4)
                spread = case.get("cluster_spread", 0.0)
                condition = case.get("condition", 1e3)
                if not isinstance(cluster, int) or cluster < 1 or not 0 <= spread < 1 or condition < 1:
                    raise ValueError("invalid cluster size, spread or condition")
                centers = np.geomspace(1, 1 / condition, (r + cluster - 1) // cluster)
                s = np.repeat(centers, cluster)[:r]
                if spread:
                    s *= np.tile(np.linspace(1 + spread / 2, 1 - spread / 2, cluster), len(centers))[:r]
                    s = np.sort(s)[::-1]
            elif family == "low_rank":
                k = case.get("rank", max(1, r // 8))
                s = np.zeros(r)
                s[:k] = np.geomspace(1, 0.1, k)
                s[k:] = case.get("tail", 1e-5)
            else:
                raise ValueError(f"Unknown matrix family: {family}")
            a = (q * s) @ v.conj().T
        matrices.append(a.astype(dtype))
    a = np.stack(matrices) if b > 1 else matrices[0]
    return np.ascontiguousarray(a * case.get("scale", 1.0))


def input_for(case, corpus=None):
    path = Path(corpus) / (case["name"] + ".npy") if corpus else None
    if path and path.exists():
        a = np.load(path, allow_pickle=False)
        shape = (case["m"], case["n"])
        if case.get("batch", 1) > 1:
            shape = (case["batch"],) + shape
        if a.shape != shape or a.dtype != np.dtype(case.get("dtype", "float32")):
            raise ValueError(f"Corpus shape/dtype mismatch: {path}")
    else:
        a = make_matrix(case)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, a, allow_pickle=False)
    digest = hashlib.sha256(a.tobytes(order="C")).hexdigest()
    return a, digest
