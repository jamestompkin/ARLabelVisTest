"""Port of `external/matlab_rgd/rdg_ADMM.m` — all three regularizers.

  - `reg="D"`   Dirichlet regularizer. Default, used by Lana's pipeline.
  - `reg="H"`   Hessian (Stein et al. 2020) regularizer. Requires an
                externally-provided `Ww_s` (the "curved hessian" matrix).
                The MATLAB version uses a MEX function `curved_hessian` from
                https://github.com/odedstein/ASmoothnessEnergyWithoutBoundaryDistortionForCurvedSurfaces
                which is NOT ported here. If you want `reg="H"`, compute
                `Ww_s` in MATLAB and pass it in.
  - `reg="vfa"` Vector-field alignment. Requires a non-zero `vf` (nf,3)
                per-face vector field; `Ww_s` is derived from it.

Variable names + stopping criteria match the MATLAB source line-for-line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, eye as sparse_eye, diags
from scipy.sparse.linalg import factorized, spsolve

from arlabelvis.rgd.mesh_ops import MeshOps

# Prefer cholespy (CHOLMOD via SuiteSparse) for SPD factor+solve. ~2x faster
# per backsolve than scipy's SuperLU on the ~1K-vertex meshes the pipeline
# produces. Ships pre-built Windows wheels via PyPI, installs with `uv add
# cholespy`. Falls back to scipy `factorized` if cholespy isn't present.
try:
    import cholespy as _cholespy
    import torch as _torch
    _HAS_CHOLESPY = True
except ImportError:
    _HAS_CHOLESPY = False


def _build_spd_solver(A_csc: csc_matrix):
    """Return a `solve(b) -> x` callable for SPD matrix `A_csc`.

    Uses cholespy/CHOLMOD when available, else scipy SuperLU. The returned
    closure owns torch buffers (cholespy) or a SuperLU object (scipy) keyed
    to the factored matrix; create one solver per factor, then call it once
    per ADMM iteration.
    """
    if _HAS_CHOLESPY:
        coo = A_csc.tocoo()
        n = A_csc.shape[0]
        ii = _torch.from_numpy(coo.row.astype(np.int32))
        jj = _torch.from_numpy(coo.col.astype(np.int32))
        vv = _torch.from_numpy(coo.data.astype(np.float64))
        solver = _cholespy.CholeskySolverD(n, ii, jj, vv, _cholespy.MatrixType.COO)
        b_buf = _torch.empty(n, dtype=_torch.float64)
        x_buf = _torch.empty(n, dtype=_torch.float64)
        b_np_view = b_buf.numpy()  # shares storage with b_buf
        x_np_view = x_buf.numpy()

        def solve_fn(b: np.ndarray) -> np.ndarray:
            np.copyto(b_np_view, b, casting="unsafe")
            solver.solve(b_buf, x_buf)
            return x_np_view.copy()  # decouple from x_buf; next solve() overwrites it

        return solve_fn
    return factorized(A_csc)


@dataclass
class AdmmHistory:
    r_norm: list = field(default_factory=list)
    s_norm: list = field(default_factory=list)
    eps_pri: list = field(default_factory=list)
    eps_dual: list = field(default_factory=list)
    rho: list = field(default_factory=list)
    iters: int = 0
    converged: bool = False


def _build_vfa_ww_s(G: csr_matrix, ta: np.ndarray, vf: np.ndarray,
                    beta: float, nf: int) -> csr_matrix:
    """Build the vector-field alignment smoothing operator Ww_s.

    Matches `rdg_ADMM.m` lines 69-72:

        Vmat[i,j] block = diag(vf[:, i] .* vf[:, j])   for i,j in {0,1,2}
        Ww_s = G' @ diag(ta, ta, ta) @ (I + beta * Vmat) @ G
    """
    rows, cols, vals = [], [], []
    for i in range(3):
        for j in range(3):
            diag_vals = vf[:, i] * vf[:, j]
            block_rows = np.arange(nf) + i * nf
            block_cols = np.arange(nf) + j * nf
            rows.append(block_rows)
            cols.append(block_cols)
            vals.append(diag_vals)
    Vmat = csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * nf, 3 * nf))
    inner = sparse_eye(3 * nf, format="csr") + beta * Vmat
    ta_diag = diags(np.tile(ta, 3))
    Ww_s = (G.T @ ta_diag @ inner @ G).tocsr()
    return Ww_s


def rdg_admm(
    mesh: MeshOps,
    x0: int,
    alpha_hat: float = 0.1,
    *,
    reg: str = "D",
    Ww_s: Optional[csr_matrix] = None,
    vf: Optional[np.ndarray] = None,
    beta_hat: float = 0.0,
    niter: int = 10000,
    abstol: float = 0.5e-5,
    reltol: float = 1e-2,
    mu: float = 10.0,
    tauinc: float = 2.0,
    taudec: float = 2.0,
    alphak: float = 1.7,
    record_history: bool = False,
) -> tuple[np.ndarray, AdmmHistory]:
    """Regularized geodesic distance from source `x0` via ADMM.

    Args:
        mesh:       MeshOps with Ww, G, va, ta precomputed.
        x0:         source vertex index (0-indexed).
        alpha_hat:  regularizer weight (scale invariant).
        reg:        "D" | "H" | "vfa".
        Ww_s:       (nv, nv) sparse smoothing operator for reg="H".
                    Required; not synthesized (needs curved_hessian MEX).
        vf:         (nf, 3) per-face vector field for reg="vfa". Required.
        beta_hat:   alignment weight for reg="vfa".
        niter, abstol, reltol, mu, tauinc, taudec, alphak:
                    ADMM hyperparameters; defaults match rdg_ADMM.m.
    Returns:
        u:         (nv,) vertex values with u[x0] == 0 and u >= 0.
        history:   residual trace (populated only if record_history=True).
    """
    nv = mesh.nv
    nf = mesh.nf
    va = mesh.va
    ta = mesh.ta
    G = mesh.G
    Ww = mesh.Ww
    sum_va = va.sum()

    # Regularizer-specific alpha, varRho, ABSTOL/RELTOL (rdg_ADMM.m lines 49-76, 91-94).
    if reg == "D":
        alpha = alpha_hat * np.sqrt(sum_va)
        var_rho = True
    elif reg == "H":
        alpha = alpha_hat * np.sqrt(sum_va ** 3)
        var_rho = False                          # line 62: varRho = 0
        abstol = abstol / 20.0                   # line 92
        reltol = reltol / 20.0                   # line 93
        if Ww_s is None:
            raise NotImplementedError(
                "reg='H' requires an externally-provided Ww_s (the Stein 2020 curved "
                "hessian). Compute it in MATLAB (`curved_hessian(V, F)`) or via libigl "
                "and pass it via Ww_s=<csr_matrix of shape (nv, nv)>."
            )
    elif reg == "vfa":
        alpha = alpha_hat * np.sqrt(sum_va)
        beta = beta_hat * np.sqrt(sum_va)
        var_rho = False                          # line 73: varRho = 0
        if vf is None:
            raise ValueError("reg='vfa' requires a per-face vector field vf of shape (nf, 3).")
        if np.linalg.norm(vf, axis=1).max() < 1e-10:
            raise ValueError("reg='vfa' vf is empty (max per-face norm < 1e-10).")
        Ww_s = _build_vfa_ww_s(G, ta, np.asarray(vf, dtype=float), beta, nf)
    else:
        raise ValueError(f"Unknown reg={reg!r}; expected 'D', 'H', or 'vfa'.")

    rho = 2.0 * np.sqrt(sum_va)                    # line 81
    thresh1 = np.sqrt(3 * nf) * abstol * np.sqrt(sum_va)   # line 96
    thresh2 = np.sqrt(nv) * abstol * sum_va                # line 97

    # Eliminate x0 from the system (lines 113-125).
    keep = np.ones(nv, dtype=bool)
    keep[x0] = False
    nv_p = np.flatnonzero(keep)

    va_p = va[keep]
    Ww_p = Ww[keep][:, keep].tocsc()
    G_p = G[:, keep].tocsr()
    G_pt = G_p.T.tocsr()
    ta_tiled = np.tile(ta, 3)
    div_p = (G_pt @ diags(ta_tiled)).tocsr()
    Ww_s_p = None
    if reg in {"H", "vfa"}:
        Ww_s_p = Ww_s[keep][:, keep].tocsc()

    # Pre-factorization (lines 135-142). For 'D' and for 'H'/'vfa' with varRho=0
    # we can factor once; for 'H' with varRho=1 (unusual) we refactor per-iter.
    # `_build_spd_solver` picks CHOLMOD if cholespy is installed, else scipy SuperLU.
    solve = None
    if reg == "D":
        solve = _build_spd_solver(Ww_p)
    elif not var_rho:   # 'H' or 'vfa'
        solve = _build_spd_solver((alpha * Ww_s_p + rho * Ww_p).tocsc())

    # Variables (lines 100-104).
    u_p = np.zeros(nv - 1)
    y = np.zeros(3 * nf)
    z = np.zeros(3 * nf)
    div_y = np.zeros(nv - 1)
    div_z = np.zeros(nv - 1)

    history = AdmmHistory()
    sqrt_ta_tiled = np.sqrt(ta_tiled)

    for ii in range(niter):
        # step 1 - u-minimization (lines 147-158)
        b = va_p - div_y + rho * div_z
        if reg == "D":
            u_p = solve(b) / (alpha + rho)
        elif not var_rho:  # 'H' or 'vfa' with varRho=0
            u_p = solve(b)
        else:              # 'H' with varRho=True: refactor each iter
            u_p = spsolve((alpha * Ww_s_p + rho * Ww_p).tocsc(), b)

        Gx = G_p @ u_p

        # step 2 - z-minimization (lines 161-170): unit-ball projection per face.
        zold = z
        div_zold = div_z
        z = (1.0 / rho) * y + Gx
        z_mat = z.reshape(3, nf).T
        norm_z = np.linalg.norm(z_mat, axis=1)
        shrink = np.where(norm_z < 1.0, 1.0, norm_z)
        z_mat = z_mat / shrink[:, None]
        z = z_mat.T.reshape(-1)
        div_z = div_p @ z

        # step 3 - dual update with over-relaxation (lines 172-174).
        y = y + rho * (alphak * Gx + (1 - alphak) * zold - z)
        div_y = div_p @ y

        # residuals (lines 176-182).
        tasqGx = sqrt_ta_tiled * Gx
        tasqZ = sqrt_ta_tiled * z
        r_norm = np.linalg.norm(tasqGx - tasqZ)
        s_norm = rho * np.linalg.norm(div_z - div_zold)
        eps_pri = thresh1 + reltol * max(np.linalg.norm(tasqGx), np.linalg.norm(tasqZ))
        eps_dual = thresh2 + reltol * np.linalg.norm(div_y)

        if record_history:
            history.r_norm.append(r_norm)
            history.s_norm.append(s_norm)
            history.eps_pri.append(eps_pri)
            history.eps_dual.append(eps_dual)
            history.rho.append(rho)

        # stopping criterion (lines 190-194).
        if ii >= 1 and r_norm < eps_pri and s_norm < eps_dual:
            history.converged = True
            history.iters = ii + 1
            break

        # varying penalty (lines 197-202). Only for reg='D' (and the 'H' varRho=1 branch).
        if var_rho:
            if r_norm > mu * s_norm:
                rho = tauinc * rho
            elif s_norm > mu * r_norm:
                rho = rho / taudec
    else:
        history.iters = niter

    u = np.zeros(nv)
    u[nv_p] = u_p
    return u, history
