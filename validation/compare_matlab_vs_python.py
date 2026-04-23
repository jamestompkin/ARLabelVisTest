# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "trimesh"]
# ///
"""Operator-level and solver-level diffs: MATLAB vs Python references for the RGD port."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # so 'arlabelvis' imports when this is run stand-alone


def tier0_operators():
    """Operator-level: ta, va, Nf, G, Ww on the icosphere must match MATLAB."""
    import trimesh
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    from scipy.sparse import csr_matrix

    # Same icosphere validate_rgd_toy.py writes (subdivisions=3, 642 verts, 1280 faces)
    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    ops = build_mesh_ops(V, F)

    ta_m = np.loadtxt(HERE / "matlab_ta.csv")
    va_m = np.loadtxt(HERE / "matlab_va.csv")
    Nf_m = np.loadtxt(HERE / "matlab_Nf.csv", delimiter=",")
    nv_nf = np.loadtxt(HERE / "matlab_nv_nf.csv", dtype=int)
    assert tuple(nv_nf) == (ops.nv, ops.nf), (nv_nf, ops.nv, ops.nf)

    # G is sparse; MATLAB dumped COO (1-indexed). Reconstruct.
    coo = np.loadtxt(HERE / "matlab_G_coo.csv", delimiter=",")
    gi = coo[:, 0].astype(int) - 1
    gj = coo[:, 1].astype(int) - 1
    gv = coo[:, 2]
    G_m = csr_matrix((gv, (gi, gj)), shape=(3 * ops.nf, ops.nv))

    print("TIER 0 - operator diffs on the icosphere")
    checks = [
        ("ta (face areas)",       np.abs(ops.ta - ta_m).max(), 1e-12),
        ("va (vertex areas)",     np.abs(ops.va - va_m).max(), 1e-12),
        ("Nf (unit face normals)", np.abs(ops.Nf - Nf_m).max(), 1e-12),
        ("G (gradient, dense diff)",
            float(np.abs(ops.G.toarray() - G_m.toarray()).max()),
            1e-10),
    ]
    any_fail = False
    for name, err, tol in checks:
        status = "PASS" if err < tol else "FAIL"
        if err >= tol:
            any_fail = True
        print(f"  {status}  {name:30s}  max|err|={err:.3e}  (tol {tol:.0e})")
    return not any_fail


def tier1_laplacian():
    py = np.loadtxt(HERE / "reference_cot_laplacian_dense.csv", delimiter=",")
    mat = np.loadtxt(HERE / "matlab_cot_laplacian_dense.csv", delimiter=",")
    assert py.shape == mat.shape, (py.shape, mat.shape)

    diff = py - mat
    max_abs = np.abs(diff).max()
    fro = np.linalg.norm(diff)
    fro_rel = fro / max(np.linalg.norm(mat), 1e-300)
    print("TIER 1 — cotangent Laplacian")
    print(f"  shape        : {py.shape}")
    print(f"  max |py-mat| : {max_abs:.3e}")
    print(f"  Frobenius    : {fro:.3e}   (rel: {fro_rel:.3e})")
    # Use relative Frobenius error as the acceptance criterion: the 4e-10 max abs
    # is just float accumulation in cos/sin; relative is effectively machine epsilon.
    passed = fro_rel < 1e-9
    print(f"  PASS bar = rel Frobenius < 1e-9  ->  {'PASS' if passed else 'FAIL'}")
    return passed


def tier2_rgd_convergence():
    """RGD converges to the discrete geodesic as alpha_hat -> 0.

    The analytic comparison is vs the unit-sphere great-circle distance. Two
    sources of expected difference:
      - Regularizer pulls u toward zero with strength alpha_hat: u shrinks in
        magnitude as alpha_hat grows. Raw L2 error is dominated by this scale.
      - Even at alpha_hat -> 0, RGD is a *discrete* variational solver on the
        mesh; it converges to analytic at a rate set by mesh resolution, not
        exactly. A 642-vertex icosphere leaves a residual shape error ~1%.
    So we report both absolute and shape-normalized errors, and expect:
      - argmax == analytic antipode at every alpha_hat.
      - raw error monotonically grows with alpha_hat (regularization strength).
      - shape-normalized error approaches a mesh-discretization floor as
        alpha_hat -> 0.
    """
    d_analytic = np.loadtxt(HERE / "reference_analytic_geodesic.csv", delimiter=",")

    files = sorted(HERE.glob("matlab_u_alphahat_*.csv"),
                   key=lambda p: float(p.stem.split("_")[-1].replace("p", ".")))
    if not files:
        print("TIER 2 - no matlab_u_alphahat_*.csv files found")
        return False

    analytic_max = float(d_analytic.max())  # = pi for a unit sphere
    analytic_am = int(np.argmax(d_analytic))
    a_n = d_analytic / analytic_max
    a_norm = float(np.linalg.norm(d_analytic))

    print("TIER 2 - RGD on the icosphere vs analytic geodesic (unit sphere)")
    print(f"  analytic max d = pi = {analytic_max:.6f}    antipode vertex = {analytic_am}")
    print()
    print(f"  {'alpha_hat':>10s}  {'max(u)':>8s}  {'raw rel L2':>11s}  "
          f"{'shape rel L2':>13s}  argmax")

    prev_shape = None
    shape_monotone_increasing = True  # shape-error should grow with alpha_hat
    argmax_ok = True
    rows = []
    for f in files:
        u = np.loadtxt(f, delimiter=",")
        alpha = float(f.stem.split("_")[-1].replace("p", "."))
        am = int(np.argmax(u))

        raw_rel = float(np.linalg.norm(u - d_analytic) / a_norm)
        u_n = u / u.max()
        shape_rel = float(np.linalg.norm(u_n - a_n) / np.linalg.norm(a_n))

        if prev_shape is not None and shape_rel <= prev_shape - 1e-6:
            shape_monotone_increasing = False
        prev_shape = shape_rel

        if am != analytic_am:
            argmax_ok = False

        rows.append((alpha, float(u.max()), raw_rel, shape_rel, am))
        print(f"  {alpha:>10.4g}  {u.max():>8.4f}  {raw_rel:>11.4f}  "
              f"{shape_rel:>13.4f}  {am}")

    shape_small = rows[0][3]   # at smallest alpha_hat

    print()
    print(f"  argmax matches analytic at every alpha_hat:    {argmax_ok}")
    print(f"  shape error monotonically grows with alpha_hat: {shape_monotone_increasing}")
    print(f"  shape error floor at alpha_hat={rows[0][0]:g}:      {shape_small:.4f}   "
          f"(heat-method O(h^2) discretization on a 642-vertex icosphere)")

    # Acceptance: argmax match + monotonicity + shape floor within mesh-induced bound.
    passed = argmax_ok and shape_monotone_increasing and shape_small < 0.02
    print(f"  PASS = argmax_ok & monotone & shape_floor<0.02  ->  {'PASS' if passed else 'FAIL'}")
    return passed


def tier1b_python_rgd_admm_vs_matlab():
    """Run the Python rdg_admm port on the icosphere and compare to MATLAB's
    u_alphahat_*.csv across the alpha_hat sweep.
    """
    import trimesh
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    from arlabelvis.rgd.admm import rdg_admm

    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    ops = build_mesh_ops(V, F)

    files = sorted(
        HERE.glob("matlab_u_alphahat_*.csv"),
        key=lambda p: float(p.stem.split("_")[-1].replace("p", ".")),
    )
    if not files:
        print("TIER 1b - no matlab_u_alphahat_*.csv present")
        return False

    print("TIER 1b - Python rdg_admm vs MATLAB on the icosphere")
    print(f"  {'alpha_hat':>10s}  {'max|u_py-u_mat|':>16s}  {'rel L2':>10s}  "
          f"{'iters_py':>10s}  argmax_py  argmax_mat")

    any_fail = False
    for f in files:
        alpha_str = f.stem.split("_")[-1]
        alpha_hat = float(alpha_str.replace("p", "."))
        u_mat = np.loadtxt(f, delimiter=",")

        u_py, hist = rdg_admm(ops, x0=0, alpha_hat=alpha_hat)

        max_err = float(np.abs(u_py - u_mat).max())
        rel_l2 = float(np.linalg.norm(u_py - u_mat) / max(np.linalg.norm(u_mat), 1e-300))
        argmax_py = int(np.argmax(u_py))
        argmax_mat = int(np.argmax(u_mat))
        iters = hist.iters

        # Acceptance: rel L2 < 1e-3 and argmax match
        ok = rel_l2 < 1e-3 and argmax_py == argmax_mat
        if not ok:
            any_fail = True
        status = "PASS" if ok else "FAIL"
        print(f"  {alpha_hat:>10.4g}  {max_err:>16.3e}  {rel_l2:>10.3e}  "
              f"{iters:>10d}  {argmax_py:>9d}  {argmax_mat:>10d}  [{status}]")

    return not any_fail


def tier1c_python_vfa_vs_matlab():
    """vfa regularizer: Python rdg_admm(reg='vfa', vf=...) vs MATLAB gold."""
    import trimesh
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    from arlabelvis.rgd.admm import rdg_admm

    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    ops = build_mesh_ops(V, F)

    vf_file = HERE / "vfa_test_vf.csv"
    gold_file = HERE / "matlab_u_vfa_alpha_p25_beta_p5.csv"
    if not (vf_file.exists() and gold_file.exists()):
        print("TIER 1c - vfa gold not present (run validation/make_vfa_test_data.py "
              "and validation/run_matlab_vfa.m first)")
        return False

    vf = np.loadtxt(vf_file, delimiter=",")
    u_mat = np.loadtxt(gold_file, delimiter=",")

    u_py, hist = rdg_admm(ops, x0=0, reg="vfa",
                          alpha_hat=0.25, beta_hat=0.5, vf=vf)
    rel_l2 = float(np.linalg.norm(u_py - u_mat) / np.linalg.norm(u_mat))
    max_err = float(np.abs(u_py - u_mat).max())
    am_py = int(np.argmax(u_py))
    am_mat = int(np.argmax(u_mat))

    ok = rel_l2 < 1e-6 and am_py == am_mat
    print("TIER 1c - Python rdg_admm(reg='vfa') vs MATLAB on the icosphere")
    print(f"  alpha_hat=0.25  beta_hat=0.5  iters_py={hist.iters}")
    print(f"  max|err|={max_err:.3e}  rel L2={rel_l2:.3e}  "
          f"argmax_py={am_py}  argmax_mat={am_mat}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def tier1d_edge_ops():
    """Per-face tangent basis + rotation + godf operator vs MATLAB dumps."""
    import trimesh
    from scipy.sparse import csr_matrix
    from arlabelvis.rgd.mesh_ops import face_normals_and_areas
    from arlabelvis.rgd.edge_ops import (
        rotation_operator, edge_basis, edge_adjacency, edge_areas, godf,
    )

    needed = ["matlab_F1.csv", "matlab_F2.csv", "matlab_R_coo.csv",
              "matlab_EB_coo.csv", "matlab_godf2_coo.csv"]
    if any(not (HERE / n).exists() for n in needed):
        print("TIER 1d - edge-op MATLAB dumps missing (run validation/run_matlab_edge_ops_dump.m)")
        return False

    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    nf = F.shape[0]
    Nf, ta = face_normals_and_areas(V, F)
    F1, F2, EB, EBI = edge_basis(V, F, Nf)
    R = rotation_operator(Nf)
    edges, e2t, ie, inner_edges = edge_adjacency(F)
    ea = edge_areas(ta, e2t, ie)
    op, oph = godf(V, F, Nf, F1, F2, edges, e2t, ie, inner_edges, ea, 2)

    F1_m = np.loadtxt(HERE / "matlab_F1.csv", delimiter=",")
    F2_m = np.loadtxt(HERE / "matlab_F2.csv", delimiter=",")

    def load_coo(path, shape):
        coo = np.loadtxt(path, delimiter=",")
        return csr_matrix((coo[:, 2], (coo[:, 0].astype(int) - 1, coo[:, 1].astype(int) - 1)),
                          shape=shape)

    R_m = load_coo(HERE / "matlab_R_coo.csv", (3 * nf, 3 * nf))
    EB_m = load_coo(HERE / "matlab_EB_coo.csv", (2 * nf, 3 * nf))
    op_m = load_coo(HERE / "matlab_godf2_coo.csv", (2 * nf, 2 * nf))

    print("TIER 1d - edge-based operators vs MATLAB")
    checks = [
        ("F1 (per-face tangent basis 1)", float(np.abs(F1 - F1_m).max()), 1e-12),
        ("F2 (per-face tangent basis 2)", float(np.abs(F2 - F2_m).max()), 1e-12),
        ("R  (rotation operator)",         float(np.abs(R.toarray() - R_m.toarray()).max()), 1e-12),
        ("EB (edge basis projection)",     float(np.abs(EB.toarray() - EB_m.toarray()).max()), 1e-12),
        ("godf(2) (direction-field op)",   float(np.abs(op.toarray() - op_m.toarray()).max()), 1e-10),
    ]
    any_fail = False
    for name, err, tol in checks:
        status = "PASS" if err < tol else "FAIL"
        if err >= tol:
            any_fail = True
        print(f"  {status}  {name:32s}  max|err|={err:.3e}  (tol {tol:.0e})")
    return not any_fail


def tier1e_smooth_vf():
    """Python smooth_vf vs MATLAB gold (line-field, n=2)."""
    import trimesh
    from arlabelvis.rgd.smooth_vf import smooth_vf

    vf_file = HERE / "smooth_vf_input.csv"
    gold_file = HERE / "matlab_smooth_vf_output.csv"
    if not (vf_file.exists() and gold_file.exists()):
        print("TIER 1e - smooth_vf gold not present "
              "(run validation/make_smooth_vf_input.py and validation/run_matlab_smooth_vf.m)")
        return False

    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    vf_in = np.loadtxt(vf_file, delimiter=",")
    w_mat = np.loadtxt(gold_file, delimiter=",")

    w_py = smooth_vf(V, F, vf_in, n=2)
    # Line-field has ±symmetry; align signs per-face.
    sign_flip = np.sign(np.sum(w_py * w_mat, axis=1, keepdims=True))
    sign_flip = np.where(sign_flip == 0, 1.0, sign_flip)
    w_py_aligned = w_py * sign_flip

    rel_fro = float(np.linalg.norm(w_py_aligned - w_mat) / np.linalg.norm(w_mat))
    max_err = float(np.abs(w_py_aligned - w_mat).max())
    dot_min = float(np.sum(w_py * w_mat, axis=1).min())
    dot_max = float(np.sum(w_py * w_mat, axis=1).max())
    # Every row should be parallel (|dot| close to 1) — a line-field equivalence.
    parallel = (np.abs(np.sum(w_py * w_mat, axis=1)) > 1 - 1e-6).all()
    ok = rel_fro < 1e-8 and parallel

    print("TIER 1e - Python smooth_vf vs MATLAB (line field, n=2)")
    print(f"  max|err| (sign-aligned) = {max_err:.3e}")
    print(f"  rel Frobenius           = {rel_fro:.3e}")
    print(f"  per-row dot prod: min={dot_min:.6f}  max={dot_max:.6f}  (all |dot|~=1 means parallel)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def tier1f_curved_hessian_pipeline():
    """reg='H' end-to-end smoke test with the real libigl curved Hessian.

    No MATLAB gold here: curved_hessian MEX isn't compiled on this install
    (Stein 2020 MEX build is blocked by the libigl-Jan2020 FindMATLAB naming
    mismatch on modern MATLAB). The Python implementation uses
    `igl.curved_hessian_energy`, which IS the same C++ code — the paper's
    author later contributed it to libigl. So the Python side has the real
    algorithm; we just validate that it composes correctly with our solver.

    Checks:
      - Q = curved_hessian_energy(V, F) is square nv x nv, symmetric, PSD.
      - rdg_admm(reg='H', Ww_s=Q) converges at multiple alpha_hats.
      - u[source]=0, u>=0, argmax is the antipode of the icosphere.
    """
    import trimesh
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    from arlabelvis.rgd.admm import rdg_admm
    try:
        from arlabelvis.rgd.curved_hessian import curved_hessian_energy
    except ImportError as e:
        print(f"TIER 1f - libigl unavailable ({e}); skipping")
        return False

    mesh = trimesh.creation.icosphere(subdivisions=3)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    ops = build_mesh_ops(V, F)

    Q = curved_hessian_energy(V, F)
    sym = (Q - Q.T).toarray()
    sym_err = float(np.abs(sym).max())
    # PSD: smallest eigval >= ~0
    import numpy.linalg as la
    evals = la.eigvalsh(Q.toarray())
    psd_ok = evals.min() > -1e-8

    print("TIER 1f - reg='H' pipeline with libigl curved Hessian")
    print(f"  Q: shape={Q.shape}  nnz={Q.nnz}  |Q-Q^T| max={sym_err:.3e}  "
          f"eigvals [{evals.min():.3e}, {evals.max():.3e}]")

    all_ok = True
    for alpha in (0.001, 0.05, 1.0):
        u, hist = rdg_admm(ops, x0=0, reg="H", alpha_hat=alpha, Ww_s=Q)
        ok = (u[0] == 0.0 and u.min() >= 0 and hist.converged and np.argmax(u) == 3)
        print(f"  alpha_hat={alpha:>6g}: iters={hist.iters:3d}  "
              f"u[src]={u[0]:.3e}  min(u)={u.min():.3e}  "
              f"argmax={int(np.argmax(u))}  converged={hist.converged}  [{'PASS' if ok else 'FAIL'}]")
        all_ok = all_ok and ok

    status = (sym_err < 1e-10) and psd_ok and all_ok
    print(f"  {'PASS' if status else 'FAIL'}")
    return status


def main():
    t0 = tier0_operators()
    print()
    t1 = tier1_laplacian()
    print()
    t1b = tier1b_python_rgd_admm_vs_matlab()
    print()
    t1c = tier1c_python_vfa_vs_matlab()
    print()
    t1d = tier1d_edge_ops()
    print()
    t1e = tier1e_smooth_vf()
    print()
    t1f = tier1f_curved_hessian_pipeline()
    print()
    t2 = tier2_rgd_convergence()
    print()
    print("SUMMARY:")
    print(f"  Tier 0  (operators):              {'PASS' if t0 else 'FAIL'}")
    print(f"  Tier 1  (cot Laplacian):          {'PASS' if t1 else 'FAIL'}")
    print(f"  Tier 1b (Python D-reg vs MATLAB): {'PASS' if t1b else 'FAIL'}")
    print(f"  Tier 1c (vfa port vs MATLAB):     {'PASS' if t1c else 'FAIL'}")
    print(f"  Tier 1d (edge ops vs MATLAB):     {'PASS' if t1d else 'FAIL'}")
    print(f"  Tier 1e (smooth_vf vs MATLAB):    {'PASS' if t1e else 'FAIL'}")
    print(f"  Tier 1f (H pipeline via libigl):  {'PASS' if t1f else 'FAIL'}")
    print(f"  Tier 2  (RGD vs analytic):        {'PASS' if t2 else 'FAIL'}")


if __name__ == "__main__":
    main()
