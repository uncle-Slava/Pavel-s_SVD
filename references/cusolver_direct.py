"""
Direct ctypes bindings to libcusolver — the routines torch does not expose.

    gesvdj_batched(A)        cusolverDn<t>gesvdjBatched   (documented n <= 32)
    gesvda_strided(A, rank)  cusolverDn<t>gesvdaStridedBatched
    gesvdp(A)                cusolverDnXgesvdp  (one matrix per call; looped)
    gesvdr(A, k, p, niters)  cusolverDnXgesvdr  (one matrix per call; looped)
    syev_batched(G)          cusolverDnXsyevBatched (CUDA >= 12.6, optional)

Memory layout: cuSOLVER is column-major.  A torch tensor A (b, m, n) is
passed as A.transpose(-1,-2).contiguous(), whose per-matrix memory *is*
column-major A with lda = m.  Outputs come back the same way and are
transposed back.  All device memory is torch tensors; the handle runs on
torch's current stream.

Written against the cuSOLVER 12.x signatures; untested here (no GPU in the
sandbox) — run `python -m gpubench.cusolver_direct` on the V100 box first:
it checks every binding against torch.linalg.svd and prints the status
codes, which is also how the n <= 32 limit of gesvdjBatched gets verified.
"""
from __future__ import annotations
import ctypes, ctypes.util, glob, os, sys
import torch

# cudaDataType
CUDA_R_32F, CUDA_R_64F, CUDA_R_16F, CUDA_C_32F, CUDA_C_64F = 0, 1, 2, 4, 5
EIG_NOVECTOR, EIG_VECTOR = 0, 1
FILL_LOWER, FILL_UPPER = 0, 1
STATUS = {0: "SUCCESS", 1: "NOT_INITIALIZED", 2: "ALLOC_FAILED", 3: "INVALID_VALUE", 4: "ARCH_MISMATCH",
          5: "MAPPING_ERROR", 6: "EXECUTION_FAILED", 7: "INTERNAL_ERROR", 8: "MATRIX_TYPE_NOT_SUPPORTED",
          9: "NOT_SUPPORTED", 10: "ZERO_PIVOT", 11: "INVALID_LICENSE"}

_lib = None
_handle = None
_params = None


class CusolverError(RuntimeError):
    pass


def _check(st, what):
    if st != 0:
        raise CusolverError(f"{what}: status {st} ({STATUS.get(st, '?')})")


def _find_lib():
    cands = []
    try:
        import nvidia.cusolver as nc                                     # pip nvidia-cusolver-cu12
        cands += glob.glob(os.path.join(os.path.dirname(nc.__file__), "lib", "libcusolver.so*"))
    except Exception:
        pass
    if os.environ.get("SVDBENCH_CUSOLVER_LIB"):                         # explicit override
        cands.insert(0, os.environ["SVDBENCH_CUSOLVER_LIB"])
    # CUDA-13 pip wheels: site-packages/nvidia/cu13/lib (torch loads libcusolver from there)
    for sp in sys.path:
        cands += glob.glob(os.path.join(sp, "nvidia", "cu13", "lib", "libcusolver.so*"))
    cands += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcusolver.so*"))
    for v in ("11", "12", "10", ""):
        name = ctypes.util.find_library(f"cusolver") if v == "" else f"libcusolver.so.{v}"
        if name:
            cands.append(name)
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise CusolverError("libcusolver not found; import torch with CUDA first, or pip install nvidia-cusolver-cu12")


def lib():
    global _lib, _handle, _params
    if _lib is None:
        assert torch.cuda.is_available(), "cusolver_direct needs CUDA"
        _lib = _find_lib()
        h = ctypes.c_void_p()
        _check(_lib.cusolverDnCreate(ctypes.byref(h)), "cusolverDnCreate")
        _handle = h
        p = ctypes.c_void_p()
        _check(_lib.cusolverDnCreateParams(ctypes.byref(p)), "cusolverDnCreateParams")
        _params = p
    _check(_lib.cusolverDnSetStream(_handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)), "SetStream")
    return _lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def _dtype_codes(dt):
    return {torch.float32: ("S", CUDA_R_32F, torch.float32), torch.float64: ("D", CUDA_R_64F, torch.float64),
            torch.complex64: ("C", CUDA_C_32F, torch.float32), torch.complex128: ("Z", CUDA_C_64F, torch.float64)}[dt]


def _cm(A):
    """column-major copy: memory of (b, n, m) row-major == column-major (m, n)"""
    return A.transpose(-1, -2).contiguous()


def version():
    L = lib()
    v = ctypes.c_int()
    L.cusolverGetProperty(0, ctypes.byref(v)); major = v.value
    L.cusolverGetProperty(1, ctypes.byref(v)); minor = v.value
    L.cusolverGetProperty(2, ctypes.byref(v)); patch = v.value
    return major, minor, patch


# ---------------------------------------------------------------------------
# gesvdjBatched
# ---------------------------------------------------------------------------
def gesvdj_batched(A, tol=None, max_sweeps=100, sort=True):
    """A: (b, m, n) CUDA.  Returns U (b,m,m), s (b,min), V (b,n,n) — the
    batched kernel returns full U and V (no econ option)."""
    L = lib()
    b, m, n = A.shape
    t, _, real_dt = _dtype_codes(A.dtype)
    Acm = _cm(A)
    S = torch.empty(b, min(m, n), dtype=real_dt, device=A.device)
    U = torch.empty(b, m, m, dtype=A.dtype, device=A.device)          # col-major m x m
    V = torch.empty(b, n, n, dtype=A.dtype, device=A.device)
    info = torch.zeros(b, dtype=torch.int32, device=A.device)
    params = ctypes.c_void_p()
    _check(L.cusolverDnCreateGesvdjInfo(ctypes.byref(params)), "CreateGesvdjInfo")
    if tol is not None:
        _check(L.cusolverDnXgesvdjSetTolerance(params, ctypes.c_double(tol)), "SetTolerance")
    _check(L.cusolverDnXgesvdjSetMaxSweeps(params, ctypes.c_int(max_sweeps)), "SetMaxSweeps")
    _check(L.cusolverDnXgesvdjSetSortEig(params, ctypes.c_int(int(sort))), "SetSortEig")
    lwork = ctypes.c_int()
    fb = getattr(L, f"cusolverDn{t}gesvdjBatched_bufferSize")
    _check(fb(_handle, EIG_VECTOR, m, n, _ptr(Acm), m, _ptr(S), _ptr(U), m, _ptr(V), n,
              ctypes.byref(lwork), params, b), f"{t}gesvdjBatched_bufferSize")
    work = torch.empty(lwork.value, dtype=A.dtype, device=A.device)
    f = getattr(L, f"cusolverDn{t}gesvdjBatched")
    st = f(_handle, EIG_VECTOR, m, n, _ptr(Acm), m, _ptr(S), _ptr(U), m, _ptr(V), n,
           _ptr(work), lwork.value, _ptr(info), params, b)
    L.cusolverDnDestroyGesvdjInfo(params)
    _check(st, f"{t}gesvdjBatched")
    torch.cuda.current_stream().synchronize()
    bad = int((info != 0).sum())
    out = {"U": U.transpose(-1, -2)[..., :min(m, n)], "s": S, "V": V.transpose(-1, -2)[..., :min(m, n)],
           "info_nonzero": bad}
    return out


# ---------------------------------------------------------------------------
# gesvdaStridedBatched
# ---------------------------------------------------------------------------
def gesvda_strided(A, rank=None, residual=False):
    """A: (b, m, n), m >= n.  Returns U (b,m,rank), s (b,rank), V (b,n,rank) and, with
    residual=True, h_err (b,) = ‖A − UΣVᴴ‖_F per matrix (h_RnrmF).  The residual is optional in
    cuSOLVER (NULL pointer) and costs extra GEMMs + a D2H copy: off by default so the timing
    matches what torch's driver="gesvda" pays."""
    L = lib()
    b, m, n = A.shape
    assert m >= n
    rank = rank or n
    t, _, real_dt = _dtype_codes(A.dtype)
    Acm = _cm(A)
    S = torch.empty(b, rank, dtype=real_dt, device=A.device)
    U = torch.empty(b, rank, m, dtype=A.dtype, device=A.device)       # col-major m x rank, ldu = m
    V = torch.empty(b, rank, n, dtype=A.dtype, device=A.device)       # col-major n x rank, ldv = n
    info = torch.zeros(b, dtype=torch.int32, device=A.device)
    herr = (ctypes.c_double * b)() if residual else None
    ll = ctypes.c_longlong
    lwork = ctypes.c_int()
    fb = getattr(L, f"cusolverDn{t}gesvdaStridedBatched_bufferSize")
    _check(fb(_handle, EIG_VECTOR, rank, m, n, _ptr(Acm), m, ll(m * n), _ptr(S), ll(rank),
              _ptr(U), m, ll(m * rank), _ptr(V), n, ll(n * rank), ctypes.byref(lwork), b),
           f"{t}gesvdaStridedBatched_bufferSize")
    work = torch.empty(max(lwork.value, 1), dtype=A.dtype, device=A.device)
    f = getattr(L, f"cusolverDn{t}gesvdaStridedBatched")
    st = f(_handle, EIG_VECTOR, rank, m, n, _ptr(Acm), m, ll(m * n), _ptr(S), ll(rank),
           _ptr(U), m, ll(m * rank), _ptr(V), n, ll(n * rank), _ptr(work), lwork.value, _ptr(info), herr, b)
    _check(st, f"{t}gesvdaStridedBatched")
    torch.cuda.current_stream().synchronize()
    out = {"U": U.transpose(-1, -2), "s": S, "V": V.transpose(-1, -2), "info_nonzero": int((info != 0).sum())}
    if residual:
        out["h_err"] = torch.tensor(list(herr))
    return out


# ---------------------------------------------------------------------------
# Xgesvdp (per matrix)
# ---------------------------------------------------------------------------
def gesvdp(A, econ=1):
    L = lib()
    b, m, n = A.shape
    t, code, real_dt = _dtype_codes(A.dtype)
    rcode = CUDA_R_32F if real_dt == torch.float32 else CUDA_R_64F
    k = min(m, n)
    Acm = _cm(A)
    S = torch.empty(b, k, dtype=real_dt, device=A.device)
    U = torch.empty(b, k if econ else m, m, dtype=A.dtype, device=A.device)   # col-major m x k
    V = torch.empty(b, k if econ else n, n, dtype=A.dtype, device=A.device)
    info = torch.zeros(b, dtype=torch.int32, device=A.device)
    i64 = ctypes.c_int64; sz = ctypes.c_size_t
    wd, wh = sz(), sz()
    _check(L.cusolverDnXgesvdp_bufferSize(_handle, _params, EIG_VECTOR, econ, i64(m), i64(n),
           code, _ptr(Acm[0]), i64(m), rcode, _ptr(S[0]), code, _ptr(U[0]), i64(m), code, _ptr(V[0]), i64(n),
           code, ctypes.byref(wd), ctypes.byref(wh)), "Xgesvdp_bufferSize")
    dwork = torch.empty(max(wd.value, 1), dtype=torch.uint8, device=A.device)
    hwork = (ctypes.c_uint8 * max(wh.value, 1))()
    herr = []
    for i in range(b):                                                  # no batched gesvdp exists
        e = ctypes.c_double()
        st = L.cusolverDnXgesvdp(_handle, _params, EIG_VECTOR, econ, i64(m), i64(n),
                                 code, _ptr(Acm[i]), i64(m), rcode, _ptr(S[i]), code, _ptr(U[i]), i64(m),
                                 code, _ptr(V[i]), i64(n), code, _ptr(dwork), sz(wd.value), hwork, sz(wh.value),
                                 _ptr(info[i:i + 1]), ctypes.byref(e))
        _check(st, "Xgesvdp")
        herr.append(e.value)
    torch.cuda.current_stream().synchronize()
    return {"U": U.transpose(-1, -2), "s": S, "V": V.transpose(-1, -2),
            "h_err_sigma": torch.tensor(herr), "info_nonzero": int((info != 0).sum())}


# ---------------------------------------------------------------------------
# Xgesvdr (per matrix)
# ---------------------------------------------------------------------------
def gesvdr(A, k, p=2, niters=2):
    L = lib()
    b, m, n = A.shape
    t, code, real_dt = _dtype_codes(A.dtype)
    rcode = CUDA_R_32F if real_dt == torch.float32 else CUDA_R_64F
    assert k + p <= min(m, n)
    Acm = _cm(A)
    S = torch.empty(b, min(m, n), dtype=real_dt, device=A.device)
    U = torch.empty(b, m, m, dtype=A.dtype, device=A.device)          # generous: m x m, ldu = m
    V = torch.empty(b, n, n, dtype=A.dtype, device=A.device)
    info = torch.zeros(b, dtype=torch.int32, device=A.device)
    i64 = ctypes.c_int64; sz = ctypes.c_size_t
    jobu = ctypes.c_byte(ord("S")); jobv = ctypes.c_byte(ord("S"))
    wd, wh = sz(), sz()
    _check(L.cusolverDnXgesvdr_bufferSize(_handle, _params, jobu, jobv, i64(m), i64(n), i64(k), i64(p), i64(niters),
           code, _ptr(Acm[0]), i64(m), rcode, _ptr(S[0]), code, _ptr(U[0]), i64(m), code, _ptr(V[0]), i64(n),
           code, ctypes.byref(wd), ctypes.byref(wh)), "Xgesvdr_bufferSize")
    dwork = torch.empty(max(wd.value, 1), dtype=torch.uint8, device=A.device)
    hwork = (ctypes.c_uint8 * max(wh.value, 1))()
    for i in range(b):
        st = L.cusolverDnXgesvdr(_handle, _params, jobu, jobv, i64(m), i64(n), i64(k), i64(p), i64(niters),
                                 code, _ptr(Acm[i]), i64(m), rcode, _ptr(S[i]), code, _ptr(U[i]), i64(m),
                                 code, _ptr(V[i]), i64(n), code, _ptr(dwork), sz(wd.value), hwork, sz(wh.value),
                                 _ptr(info[i:i + 1]))
        _check(st, "Xgesvdr")
    torch.cuda.current_stream().synchronize()
    return {"U": U.transpose(-1, -2)[..., :k], "s": S[..., :k], "V": V.transpose(-1, -2)[..., :k],
            "info_nonzero": int((info != 0).sum())}


# ---------------------------------------------------------------------------
# XsyevBatched (CUDA >= 12.6)
# ---------------------------------------------------------------------------
def syev_batched(G):
    """G: (b, n, n) Hermitian.  Returns w (b, n) ascending, V (b, n, n)."""
    L = lib()
    if not hasattr(L, "cusolverDnXsyevBatched_bufferSize"):
        raise CusolverError("XsyevBatched not in this libcusolver (needs CUDA >= 12.6)")
    b, n, _ = G.shape
    t, code, real_dt = _dtype_codes(G.dtype)
    rcode = CUDA_R_32F if real_dt == torch.float32 else CUDA_R_64F
    Gcm = _cm(G)                                                        # eigenvectors overwrite A
    W = torch.empty(b, n, dtype=real_dt, device=G.device)
    info = torch.zeros(b, dtype=torch.int32, device=G.device)
    i64 = ctypes.c_int64; sz = ctypes.c_size_t
    wd, wh = sz(), sz()
    _check(L.cusolverDnXsyevBatched_bufferSize(_handle, _params, EIG_VECTOR, FILL_LOWER, i64(n), code, _ptr(Gcm), i64(n),
           rcode, _ptr(W), code, ctypes.byref(wd), ctypes.byref(wh), i64(b)), "XsyevBatched_bufferSize")
    dwork = torch.empty(max(wd.value, 1), dtype=torch.uint8, device=G.device)
    hwork = (ctypes.c_uint8 * max(wh.value, 1))()
    st = L.cusolverDnXsyevBatched(_handle, _params, EIG_VECTOR, FILL_LOWER, i64(n), code, _ptr(Gcm), i64(n),
                                  rcode, _ptr(W), code, _ptr(dwork), sz(wd.value), hwork, sz(wh.value), _ptr(info), i64(b))
    _check(st, "XsyevBatched")
    torch.cuda.current_stream().synchronize()
    return {"w": W, "V": Gcm.transpose(-1, -2), "info_nonzero": int((info != 0).sum())}


# ---------------------------------------------------------------------------
# registration into the algorithm registry
# ---------------------------------------------------------------------------
class _Wrap:
    """Registry entry: get_algorithm() instantiates classes with **kw, so each
    binding is wrapped in its own small class (see _make)."""
    fn = None

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, A, state=None):
        return type(self).fn(A, **self.kw)


def _make(fn):
    return type("Direct_" + fn.__name__, (_Wrap,), {"fn": staticmethod(fn)})


class GramXsyevBatched:
    """AᴴA (torch GEMM) → cusolverDnXsyevBatched → U = AVΣ⁻¹: the gesvda idea
    with the CUDA-12.6+ batched symmetric eigensolver instead of syevj."""
    def __init__(self, **kw):
        pass

    def __call__(self, A, state=None):
        G = A.mH @ A; G = 0.5 * (G + G.mH)
        o = syev_batched(G)
        w = o["w"].flip(-1).clamp_min(0); V = o["V"].flip(-1)
        s = w.sqrt()
        return {"U": (A @ V) / s.clamp_min(1e-30).unsqueeze(-2).to(A.dtype), "s": s, "V": V,
                "info_nonzero": o["info_nonzero"]}


def register_all():
    from .algorithms import ALGORITHMS
    # (class, needs_cuda, default kw, capturable): every binding synchronises on info/h_err
    ALGORITHMS["cusolver_gesvdjBatched"] = (_make(gesvdj_batched), True, {}, False)
    ALGORITHMS["cusolver_gesvdaStrided"] = (_make(gesvda_strided), True, {}, False)
    ALGORITHMS["cusolver_Xgesvdp"] = (_make(gesvdp), True, {}, False)
    ALGORITHMS["cusolver_Xgesvdr"] = (_make(gesvdr), True, {"k": 2, "p": 1, "niters": 2}, False)
    ALGORITHMS["cusolver_gram_XsyevBatched"] = (GramXsyevBatched, True, {}, False)


def selfcheck():
    """Run on the GPU box: verifies each binding against torch and reports
    which routines exist / which sizes they accept."""
    assert torch.cuda.is_available()
    print("cuSOLVER", version(), "| torch", torch.__version__, "| GPU", torch.cuda.get_device_name())
    dev = "cuda"
    for dt in (torch.complex64, torch.float32):
        for (m, n) in [(16, 4), (32, 32), (64, 4), (64, 64)]:
            A = torch.randn(4, m, n, dtype=dt, device=dev)
            ref = torch.linalg.svdvals(A)
            for name, fn in [("gesvdjBatched", gesvdj_batched), ("gesvdaStrided", gesvda_strided),
                             ("Xgesvdp", gesvdp), ("Xgesvdr(k=2,p=1)", lambda X: gesvdr(X, 2, 1, 2))]:
                try:
                    o = fn(A)
                    kk = o["s"].shape[-1]
                    err = (o["s"] - ref[..., :kk]).abs().max().item() / ref[..., 0].max().item()
                    print(f"  {str(dt):16s} {m:3d}x{n:<3d} {name:18s} OK   σ err {err:.1e}  info!=0: {o['info_nonzero']}")
                except CusolverError as ex:
                    print(f"  {str(dt):16s} {m:3d}x{n:<3d} {name:18s} FAIL {ex}")
                except Exception as ex:
                    print(f"  {str(dt):16s} {m:3d}x{n:<3d} {name:18s} ERROR {type(ex).__name__}: {ex}")
        G = torch.randn(4, 8, 8, dtype=dt, device=dev); G = G @ G.mH
        try:
            o = syev_batched(G)
            print(f"  {str(dt):16s} XsyevBatched OK  λ err {(o['w'] - torch.linalg.eigvalsh(G)).abs().max().item():.1e}")
        except Exception as ex:
            print(f"  {str(dt):16s} XsyevBatched unavailable: {ex}")


if __name__ == "__main__":
    selfcheck()
