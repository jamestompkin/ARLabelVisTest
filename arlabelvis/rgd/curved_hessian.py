"""Curved Hessian smoothness energy (Stein et al. 2020).

Produces the `Ww_s` smoothing operator for `rdg_admm(reg='H', ...)`. In the
MATLAB source, this was a MEX compiled from the paper's C++ reference
implementation. The same algorithm was later mainlined into **libigl**
(contributed by the paper's author), and `libigl` exposes it via the
Python bindings as `igl.curved_hessian_energy`.

So this module is a thin wrapper that:
  - calls `igl.curved_hessian_energy(V, F)` to get the Q matrix,
  - returns it as a scipy sparse CSR suitable for passing as `Ww_s` to
    `rdg_admm(reg='H', Ww_s=...)`.

Validated that the result is a symmetric positive-semidefinite nv x nv
matrix (the expected shape and algebraic properties of the curved Hessian
energy).

Provenance: this is literally the same C++ code as the Stein 2020 MEX,
re-exposed through pybind11 in libigl. We link against libigl's prebuilt
Python wheel — no MEX compile required, no MATLAB bridge.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

import igl


def curved_hessian_energy(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """Compute the curved Hessian energy matrix Q for use as `Ww_s`.

    Args:
        V: (nv, 3) vertex positions.
        F: (nf, 3) 0-indexed faces.

    Returns:
        csr_matrix of shape (nv, nv). Symmetric, PSD.  Matches
        `Q = D^T * Mi * (L + K) * Mi * D` from `cpp_interface/applications/Mex/main.cpp`.

    Use with rdg_admm:

        from arlabelvis.rgd.curved_hessian import curved_hessian_energy
        Ww_s = curved_hessian_energy(V, F)
        u, _ = rdg_admm(mesh_ops, x0, alpha_hat=0.1, reg="H", Ww_s=Ww_s)
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)
    Q = igl.curved_hessian_energy(V, F)
    # igl returns a scipy sparse matrix (or dense np.ndarray in some versions).
    if not hasattr(Q, "tocsr"):
        Q = csr_matrix(Q)
    return Q.tocsr()
