"""Port of `external/matlab_rgd/rdg_allpairs_admm.m`.

Joint ADMM that solves all-pairs regularized geodesic distances in one big
matrix ADMM rather than looping over sources. Only the Dirichlet regularizer
is implemented (MATLAB source has no other paths here).

Variables mirror MATLAB:
  X (nv, nv)   columns are distance fields, with gradient along columns
  R (nv, nv)   transposed-source variant (gradient along rows)
  U (nv, nv)   dual consensus: the all-pairs distance matrix
  Z, Q (3nf, nv)   auxiliary unit vector fields (GX = Z, GR = Q)
  Y, S (3nf, nv)   dual for Z, Q
  H, K (nv, nv)    dual for X≈U and R≈U'

Output is the final `U`. This is the distance matrix; `argmax(U, axis=0)` gives
per-source argmax, equivalent to the `max_indices` demo.m produces via the
parfor + single-source `rdg_ADMM` path.

NOTE: this solver is much more memory- and compute-hungry than the per-source
loop. For a 2562-vertex mesh it needs ~6 dense (2562, 2562) doubles (~200 MB)
plus several (3nf, nv) dense matrices. Not suitable for 50K+ vertex meshes
without batching/streaming. Lana's pipeline uses the single-source loop for
this reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu

from arlabelvis.rgd.mesh_ops import MeshOps


@dataclass
class AllPairsHistory:
    r_norm: list = field(default_factory=list)
    s_norm: list = field(default_factory=list)
    r_norm2: list = field(default_factory=list)
    s_norm2: list = field(default_factory=list)
    r_xr1: list = field(default_factory=list)
    r_xr2: list = field(default_factory=list)
    s_xr: list = field(default_factory=list)
    iters: int = 0
    converged: bool = False


def rdg_allpairs_admm(
    mesh: MeshOps,
    alpha_hat: float,
    *,
    niter: int = 20000,
    abstol: float = 1e-6,
    reltol: float = 2e-4,
    mu: float = 10.0,
    tauinc: float = 2.0,
    taudec: float = 2.0,
    alphak: float = 1.7,
    quiet: bool = True,
    record_history: bool = False,
) -> tuple[np.ndarray, AllPairsHistory]:
    """All-pairs RGD via joint ADMM. Returns U (nv, nv) distance matrix.

    Defaults (niter=20000, abstol=1e-6, reltol=2e-4) match MATLAB source.
    """
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va
    ta = mesh.ta
    G = mesh.G
    Ww = mesh.Ww

    sum_va = va.sum()
    vasq = np.sqrt(va)
    va_mat = diags(va)
    vainv = 1.0 / va
    va_outer = np.outer(va, va)          # vavatMat
    ta_tiled = np.tile(ta, 3)
    ta_mat = diags(ta_tiled)             # (3 nf, 3 nf)
    div = (G.T @ ta_mat).tocsr()         # (nv, 3 nf)
    tasq = np.sqrt(ta_tiled)

    alpha = alpha_hat * np.sqrt(sum_va)

    # Two ADMM penalties. rho for Z≈GX / Q≈GR; rho2 for X≈U / R≈U'.
    rho = 2.0 * np.sqrt(sum_va)
    rho2 = 10.0 / np.sqrt(sum_va)

    thresh1 = np.sqrt(3 * nf) * abstol * sum_va
    thresh2 = np.sqrt(nv) * abstol * (sum_va ** 2)
    thresh3 = np.sqrt(nv) * abstol * (np.sqrt(sum_va) ** 3)
    thresh4 = np.sqrt(nv) * abstol * sum_va

    # Factor the "u-step" system. Refactor when rho or rho2 changes.
    def make_lu(rho, rho2):
        A = (alpha + rho) * Ww + rho2 * va_mat
        return splu(A.tocsc())

    lu = make_lu(rho, rho2)

    # State.
    X = np.zeros((nv, nv))
    R = np.zeros((nv, nv))
    U = np.zeros((nv, nv))
    Z = np.zeros((3 * nf, nv))
    Q = np.zeros((3 * nf, nv))
    Y = np.zeros((3 * nf, nv))
    S = np.zeros((3 * nf, nv))
    H = np.zeros((nv, nv))
    K = np.zeros((nv, nv))
    div_Z = np.zeros((nv, nv))
    div_Q = np.zeros((nv, nv))
    div_Y = np.zeros((nv, nv))
    div_S = np.zeros((nv, nv))

    history = AllPairsHistory()

    # RHS constant piece:  0.5 * va_outer * vainv'   (column-wise scale of va_outer by 1/va).
    # Broadcast: va_outer shape (nv,nv); vainv[None, :] shape (1,nv) scales columns.
    const_term = 0.5 * va_outer * vainv[None, :]

    for ii in range(niter):
        # step 1 - X, R minimization (lines 98-111).
        bx = (const_term
              - div_Y + rho * div_Z
              - (va[:, None] * H)
              + rho2 * (va[:, None] * U))
        br = (const_term
              - div_S + rho * div_Q
              - (va[:, None] * K)
              + rho2 * (va[:, None] * U.T))

        X = lu.solve(bx)
        R = lu.solve(br)

        Gx = G @ X    # (3 nf, nv)
        Gr = G @ R

        # step 2 - Z, Q, U minimization (lines 114-133).
        Zold = Z
        div_Zold = div_Z
        Z = (1.0 / rho) * Y + Gx
        # Reshape to (nf, 3, nv), normalize column-wise over the 3-axis, reshape back.
        Z = Z.reshape(3, nf, nv).transpose(1, 0, 2)            # (nf, 3, nv)
        norm_Z = np.sqrt((Z ** 2).sum(axis=1, keepdims=True))  # (nf, 1, nv)
        shrink = np.where(norm_Z < 1.0, 1.0, norm_Z)
        Z = Z / shrink
        Z = Z.transpose(1, 0, 2).reshape(3 * nf, nv)
        div_Z = div @ Z

        Qold = Q
        div_Qold = div_Q
        Q = (1.0 / rho) * S + Gr
        Q = Q.reshape(3, nf, nv).transpose(1, 0, 2)
        norm_Q = np.sqrt((Q ** 2).sum(axis=1, keepdims=True))
        shrink = np.where(norm_Q < 1.0, 1.0, norm_Q)
        Q = Q / shrink
        Q = Q.transpose(1, 0, 2).reshape(3 * nf, nv)
        div_Q = div @ Q

        Uold = U
        U1 = 0.5 * ((1.0 / rho2) * (H + K.T) + X + R.T)
        # Zero the diagonal, clip negatives to 0 (distance matrix is non-negative, zero on diagonal).
        U = U1 - np.diag(np.diag(U1))
        U = np.where(U < 0, 0, U)

        # step 3 - dual variable update (lines 135-141).
        Y = Y + rho * (alphak * Gx + (1 - alphak) * Zold - Z)
        S = S + rho * (alphak * Gr + (1 - alphak) * Qold - Q)
        H = H + rho2 * (alphak * X + (1 - alphak) * Uold - U)
        K = K + rho2 * (alphak * R + (1 - alphak) * Uold.T - U.T)
        div_Y = div @ Y
        div_S = div @ S

        # Residuals (lines 143-163).
        GxW = (tasq[:, None] * Gx) * vasq[None, :]
        ZW = (tasq[:, None] * Z) * vasq[None, :]
        GrW = (tasq[:, None] * Gr) * vasq[None, :]
        QW = (tasq[:, None] * Q) * vasq[None, :]

        r_norm = np.linalg.norm(GxW - ZW)
        eps_pri = thresh1 + reltol * max(np.linalg.norm(GxW), np.linalg.norm(ZW))
        s_norm = rho * np.linalg.norm((div_Z - div_Zold) * va[None, :])
        eps_dual = thresh2 + reltol * np.linalg.norm(div_Y * va[None, :])

        r_norm2 = np.linalg.norm(GrW - QW)
        eps_pri2 = thresh1 + reltol * max(np.linalg.norm(GrW), np.linalg.norm(QW))
        s_norm2 = rho * np.linalg.norm((div_Q - div_Qold) * va[None, :])
        eps_dual2 = thresh2 + reltol * np.linalg.norm(div_S * va[None, :])

        r_xr1 = np.linalg.norm((vasq[:, None] * (X - U)) * vasq[None, :])
        r_xr2 = np.linalg.norm((vasq[:, None] * (R - U.T)) * vasq[None, :])
        eps_pri_xr1 = thresh3 + reltol * min(
            np.linalg.norm((vasq[:, None] * X) * vasq[None, :]),
            np.linalg.norm((vasq[:, None] * U) * vasq[None, :]),
        )
        eps_pri_xr2 = thresh3 + reltol * min(
            np.linalg.norm((vasq[:, None] * R) * vasq[None, :]),
            np.linalg.norm((vasq[:, None] * U.T) * vasq[None, :]),
        )
        s_xr = np.sqrt(2) * rho2 * np.linalg.norm((vasq[:, None] * (U - Uold)) * vasq[None, :])
        eps_dual_xr = thresh4 + reltol * 0.5 * (
            np.linalg.norm((vasq[:, None] * H) * vasq[None, :])
            + np.linalg.norm((vasq[:, None] * K) * vasq[None, :])
        )

        if record_history:
            history.r_norm.append(r_norm)
            history.s_norm.append(s_norm)
            history.r_norm2.append(r_norm2)
            history.s_norm2.append(s_norm2)
            history.r_xr1.append(r_xr1)
            history.r_xr2.append(r_xr2)
            history.s_xr.append(s_xr)

        if not quiet and (ii % 10) == 0:
            print(f"iter {ii:4d}  r={r_norm:.3e}/{eps_pri:.3e}  s={s_norm:.3e}/{eps_dual:.3e}  "
                  f"r2={r_norm2:.3e}/{eps_pri2:.3e}  xr1={r_xr1:.3e}/{eps_pri_xr1:.3e}  "
                  f"sxr={s_xr:.3e}/{eps_dual_xr:.3e}  rho={rho:.2f}  rho2={rho2:.2f}")

        # Stopping (lines 177-185) — all seven residuals within tolerance.
        if (r_norm < eps_pri and s_norm < eps_dual
                and r_norm2 < eps_pri2 and s_norm2 < eps_dual2
                and r_xr1 < eps_pri_xr1 and r_xr2 < eps_pri_xr2
                and s_xr < eps_dual_xr):
            history.converged = True
            history.iters = ii + 1
            break

        # Varying penalty (lines 188-207).
        rho_changed = False
        ratio1 = r_norm / max(eps_pri, 1e-30)
        ratio2 = s_norm / max(eps_dual, 1e-30)
        ratio1b = r_norm2 / max(eps_pri2, 1e-30)
        ratio2b = s_norm2 / max(eps_dual2, 1e-30)
        if ratio1 > mu * ratio2 and ratio1b > mu * ratio2b:
            rho = tauinc * rho
            rho_changed = True
        elif ratio2 > mu * ratio1 and ratio2b > mu * ratio1b:
            rho = rho / taudec
            rho_changed = True

        ratio_xr1 = r_xr1 / max(eps_pri_xr1, 1e-30)
        ratio_xr2 = r_xr2 / max(eps_pri_xr2, 1e-30)
        ratio_sxr = s_xr / max(eps_dual_xr, 1e-30)
        if ratio_xr1 > mu * ratio_sxr and ratio_xr2 > mu * ratio_sxr:
            rho2 = tauinc * rho2
            rho_changed = True
        elif ratio_sxr > mu * ratio_xr1 and ratio_sxr > mu * ratio_xr2:
            rho2 = rho2 / taudec
            rho_changed = True

        if rho_changed:
            lu = make_lu(rho, rho2)
    else:
        history.iters = niter

    return U, history
