# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Tier-1 and Tier-2 quantitative comparison: MATLAB vs Python references."""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


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
    passed = max_abs < 1e-10
    print(f"  PASS bar = 1e-10  ->  {'PASS' if passed else 'FAIL'}")
    return passed


def tier2_rgd_convergence():
    # Analytic closed-form geodesic on the unit sphere from vertex 0
    d_analytic = np.loadtxt(HERE / "reference_analytic_geodesic.csv", delimiter=",")

    # Collect all MATLAB u_alpha files
    files = sorted(HERE.glob("matlab_u_alphahat_*.csv"),
                   key=lambda p: float(p.stem.split("_")[-1].replace("p", ".")))
    if not files:
        print("TIER 2 — no matlab_u_alphahat_*.csv files found")
        return False

    print("TIER 2 — RGD convergence to analytic geodesic on unit sphere")
    print(f"  analytic max d = pi = {d_analytic.max():.6f}")
    # Normalize by max for shape comparison (RGD and true geodesic differ by a scale)
    a_n = d_analytic / d_analytic.max()

    print(f"  {'alpha_hat':>10s}  {'max(u)':>10s}  {'L2 rel (shape)':>16s}  "
          f"{'spearman-ish':>14s}  argmax")
    prev_rel = None
    monotone = True
    argmax_match = True
    for f in files:
        u = np.loadtxt(f, delimiter=",")
        # Normalize u by max to compare *shapes* (not absolute magnitudes)
        u_n = u / u.max()
        rel_l2 = np.linalg.norm(u_n - a_n) / np.linalg.norm(a_n)
        # Rank correlation (loose "spearman") — use Pearson on ranks
        ranks_u = np.argsort(np.argsort(u))
        ranks_a = np.argsort(np.argsort(d_analytic))
        pearson = np.corrcoef(ranks_u, ranks_a)[0, 1]

        alpha = float(f.stem.split("_")[-1].replace("p", "."))
        am = int(np.argmax(u))
        analytic_am = int(np.argmax(d_analytic))
        # Antipode of vertex 0 on the subdivided icosphere is vertex 3
        print(f"  {alpha:>10.4g}  {u.max():>10.4f}  {rel_l2:>16.4f}  "
              f"{pearson:>14.4f}  {am} (analytic={analytic_am})")

        if am != analytic_am:
            argmax_match = False
        if prev_rel is not None and rel_l2 > prev_rel + 1e-6:
            monotone = False
        prev_rel = rel_l2

    smallest_alpha = files[0]
    u_small = np.loadtxt(smallest_alpha, delimiter=",")
    u_n = u_small / u_small.max()
    final_rel = np.linalg.norm(u_n - a_n) / np.linalg.norm(a_n)

    print(f"\n  Convergence at alpha_hat={float(smallest_alpha.stem.split('_')[-1].replace('p', '.')):.4g}: "
          f"rel L2 shape error = {final_rel:.4f}")
    print(f"  Argmax matches analytic at all alphas: {argmax_match}")
    print(f"  Rel L2 shape error monotone in alpha: {monotone}")

    # Acceptance: at the smallest alpha, shape error < 0.15 (loose bound — RGD at
    # finite alpha is inherently smoothed vs the true geodesic)
    ok_shape = final_rel < 0.15
    passed = argmax_match and ok_shape
    print(f"  PASS = argmax_match & rel<0.15  ->  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    t1 = tier1_laplacian()
    print()
    t2 = tier2_rgd_convergence()
    print()
    print("SUMMARY:")
    print(f"  Tier 1 (cot Laplacian): {'PASS' if t1 else 'FAIL'}")
    print(f"  Tier 2 (RGD vs analytic): {'PASS' if t2 else 'FAIL'}")


if __name__ == "__main__":
    main()
