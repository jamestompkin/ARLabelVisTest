"""Port of `external/matlab_rgd/smooth_vf.m`.

Smooth a sparse per-face vector field by projecting into the power-n
representation, solving a constrained least-squares problem on the
globally-optimal-direction-field (godf) energy, and un-projecting.

Used to generate the `vf` input for `rdg_admm(reg='vfa', ...)`.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve, eigsh

from arlabelvis.rgd.mesh_ops import face_normals_and_areas
from arlabelvis.rgd.edge_ops import (
    edge_basis, edge_adjacency, edge_areas, godf,
)


def _ff(v: np.ndarray, n) -> np.ndarray:
    """Complex exponentiation used by smooth_vf's symmetry reduction.

    Interprets the first half of `v` as real parts and the second half as imag
    parts (`(a+ib)` per entry), returns `(a+ib)^n` in the same layout.
    Matches MATLAB `ff` in `smooth_vf.m`.
    """
    s = len(v) // 2
    a = v[:s]
    b = v[s:]
    c = a + 1j * b
    cn = c ** n
    return np.concatenate([cn.real, cn.imag])


def smooth_vf(V: np.ndarray, F: np.ndarray, vf_in: np.ndarray, n: int = 2) -> np.ndarray:
    """Smooth a per-face vector field aligned to `vf_in` where it is non-zero.

    Args:
        V, F:   mesh data (0-indexed faces).
        vf_in:  (nf, 3) vector field; rows with norm > 1e-5 are treated as
                alignment constraints, others are free.
        n:      symmetry order. `n=2` is a line field (unsigned direction),
                the default in MATLAB smooth_vf.

    Returns:
        w: (nf, 3) smoothed, unit-norm per-face vector field.

    Line-for-line port of `smooth_vf.m`. Solves

        min_{x in R^(2 nf)}  ||op * x||^2     s.t.  Aeq * x = beq

    where `op = godf(n)`, and (Aeq, beq) encode that `x` must equal the
    power-n tangent-basis representation of `vf_in` at constrained faces.
    """
    vf_in = np.asarray(vf_in, dtype=float)
    nf = F.shape[0]
    assert vf_in.shape == (nf, 3), f"vf_in must be (nf, 3); got {vf_in.shape}"

    # Mesh ops
    Nf, ta = face_normals_and_areas(V, F)
    F1, F2, EB, EBI = edge_basis(V, F, Nf)
    edges, e2t, ie, inner_edges = edge_adjacency(F)
    ea = edge_areas(ta, e2t, ie)
    op, oph = godf(V, F, Nf, F1, F2, edges, e2t, ie, inner_edges, ea, n)

    # Find constrained faces
    vf_norms = np.linalg.norm(vf_in, axis=1)
    locs = np.where(vf_norms > 1e-5)[0]
    nl = len(locs)

    # Project vf to 2D tangent basis: EB expects `[x; y; z]` stacked.
    vf_stacked = np.concatenate([vf_in[:, 0], vf_in[:, 1], vf_in[:, 2]])
    EB_vf = EB @ vf_stacked                        # (2 nf,)
    # ff(EB*vf, n): reinterpret as complex, raise to power n
    EB_vf_n = _ff(EB_vf, n)                        # (2 nf,)
    # MATLAB: reshape to (nf, 2) column-major, gather constrained rows,
    # then (:) column-major flatten -> (2 nl,)
    beq_full = EB_vf_n.reshape(2, nf).T            # (nf, 2)
    beq = beq_full[locs, :]                        # (nl, 2)
    beq = np.concatenate([beq[:, 0], beq[:, 1]])   # (2 nl,)

    if nl > 0:
        # Aeq: (2 nl, 2 nf). Row i selects x[locs[i]]; row nl+i selects x[nf + locs[i]].
        rows = np.arange(2 * nl)
        cols = np.concatenate([locs, nf + locs])
        Aeq = csr_matrix((np.ones(2 * nl), (rows, cols)), shape=(2 * nl, 2 * nf))

        # Equality-constrained least squares via KKT. objective ||op x||^2 -> H = op' op.
        H = (op.T @ op).tocsc()
        K = bmat([[H, Aeq.T], [Aeq, None]], format="csc")
        rhs = np.concatenate([np.zeros(2 * nf), beq])
        z = spsolve(K, rhs)
        x = z[:2 * nf]
    else:
        # Unconstrained: eigenvector of smallest eigenvalue of op (matches MATLAB's eigs(C,1,'SM'))
        _, evec = eigsh(op.tocsc(), k=1, sigma=0.0, which="LM")
        x = evec.ravel()

    # Un-power and project back to 3D
    x_un = _ff(x, 1.0 / n)                         # (2 nf,)
    w_stacked = EBI @ x_un                         # (3 nf,)
    w = w_stacked.reshape(3, nf).T                 # (nf, 3)

    # Normalize per face (rows with near-zero norm stay zero)
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    safe = np.where(norms < 1e-15, 1.0, norms)
    w = np.where(norms < 1e-15, 0.0, w / safe)
    return w
