# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "trimesh"]
# ///
"""Diff MATLAB vs Python all-pairs argmax on the deformed icosphere, with
plateau-margin analysis for any mismatches.

A mismatch is "acceptable" if MATLAB's and Python's choices are two near-tied
candidates in u (relative margin below tol) — i.e., a float-noise flip, not a
real divergence. We recompute u_python per-mismatched-source and report both:

  - max(u) - u[matlab_argmax]  (how far MATLAB's pick was from our top in our u)
  - max(u) - u[python_argmax]  (trivially 0, but prints for sanity)

If (max(u) - u[matlab_argmax]) / max(u) is tiny (~1e-8 or smaller), the two
implementations are picking between floating-point-indistinguishable siblings.
That's acceptable. If it's large, there's a real disagreement to investigate.
"""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from arlabelvis.rgd.mesh_ops import build_mesh_ops
from arlabelvis.rgd.admm import rdg_admm
from arlabelvis.rgd.allpairs import read_off


ALPHA = 0.25
MESH = HERE.parent / "external/matlab_rgd/deformed_ico_sub4.off"
M_FILE = HERE / "matlab_allpairs_deformed_alpha_p25.csv"
P_FILE = HERE / "python_allpairs_deformed_alpha_p25.txt"


def main():
    m = np.loadtxt(M_FILE, delimiter=",", dtype=int).ravel()
    p = np.loadtxt(P_FILE, delimiter=",", dtype=int).ravel()
    assert m.shape == p.shape, (m.shape, p.shape)
    n = len(m)

    agree = int((m == p).sum())
    mismatch_idx = np.where(m != p)[0]
    print(f"sources: {n}")
    print(f"exact argmax agreement: {agree}/{n}  ({100*agree/n:.3f}%)")
    print(f"mismatches:             {len(mismatch_idx)}")

    if len(mismatch_idx) == 0:
        print("\nIDENTICAL - no plateau cases needed analysis.")
        return 0

    print("\nRecomputing Python u for each mismatched source to measure plateau margin...")
    V, F = read_off(str(MESH))
    ops = build_mesh_ops(V, F)

    # MATLAB files are 1-indexed; convert to 0-indexed for u lookups.
    margins = []
    for src0 in mismatch_idx:
        u, _ = rdg_admm(ops, x0=int(src0), alpha_hat=ALPHA)
        py_argmax = int(np.argmax(u))
        mat_argmax_0 = int(m[src0]) - 1   # 1->0 indexed
        umax = float(u.max())
        u_at_mat_pick = float(u[mat_argmax_0])
        abs_gap = umax - u_at_mat_pick
        rel_gap = abs_gap / umax if umax > 0 else 0.0
        margins.append((src0, py_argmax, mat_argmax_0, umax, u_at_mat_pick, abs_gap, rel_gap))

    margins_rel = np.array([r[-1] for r in margins])
    print("\nPlateau margin distribution (how far MATLAB's pick is from the Python-u top):")
    print(f"  n mismatches:           {len(margins)}")
    print(f"  max  relative gap:      {margins_rel.max():.3e}")
    print(f"  median relative gap:    {np.median(margins_rel):.3e}")
    print(f"  mean relative gap:      {margins_rel.mean():.3e}")

    # Show a few worst-case
    worst = sorted(margins, key=lambda r: -r[-1])[:10]
    print("\n  worst 10 by relative gap:")
    print(f"  {'src':>6s}  {'py_am':>6s}  {'mat_am':>7s}  {'u.max':>10s}  "
          f"{'u[mat]':>10s}  {'abs gap':>10s}  {'rel gap':>10s}")
    for src0, py_am, mat_am, umax, u_at_mat, abs_gap, rel_gap in worst:
        print(f"  {src0:>6d}  {py_am:>6d}  {mat_am:>7d}  {umax:>10.6f}  "
              f"{u_at_mat:>10.6f}  {abs_gap:>10.3e}  {rel_gap:>10.3e}")

    # Acceptance: every mismatch is a plateau flip, relative gap < 1e-6.
    PLATEAU_TOL = 1e-6
    clean_plateaus = int((margins_rel < PLATEAU_TOL).sum())
    real_diffs = int((margins_rel >= PLATEAU_TOL).sum())
    print(f"\n  mismatches within plateau tol ({PLATEAU_TOL:g}): {clean_plateaus}")
    print(f"  real disagreements (above tol):              {real_diffs}")
    if real_diffs == 0:
        print("\nPASS - every mismatch is a plateau flip within float noise.")
        return 0
    print("\nFAIL - there are real disagreements between MATLAB and Python beyond plateau noise.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
