# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh", "scipy"]
# ///
"""
Tier-1 and tier-2 RGD validation scaffolding.

Produces a toy-mesh problem with known closed-form answers, so that when
MATLAB demo.m is run on the same .off file we can diff against ground truth
and against a Python reference cotangent Laplacian.

Outputs (all written to this folder):
  icosphere_sub3.off                 OFF mesh (642 verts, 1280 faces), unit sphere
  reference_analytic_geodesic.csv    Exact geodesic distance from vertex 0 on unit sphere
  reference_cot_laplacian.npz        Python cotangent Laplacian (MATLAB sign convention)
  reference_cot_laplacian_dense.csv  Same, dense (for eyeball diff against MATLAB)
  reference_invariants.txt           Pass/fail of self-consistency checks on the Python ref
  EXPECTED_MATLAB_OUTPUT.md          What to check once demo.m has run on the .off
"""
from __future__ import annotations
import numpy as np
import trimesh
from pathlib import Path
from scipy.sparse import csr_matrix

HERE = Path(__file__).parent


def write_off(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    """OFF format matching external/matlab_rgd/readOff.m expectations."""
    with path.open("w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in F:
            # readOff.m is 1-indexed internally; OFF on disk is 0-indexed
            # (readOff adds +1 after parsing). We write 0-indexed per spec.
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def cot_laplacian(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """Cotangent Laplacian matching external/matlab_rgd/cotLaplacian.m:
       W_ii = +sum over incident half-cotangents,  W_ij = -0.5*(cot alpha + cot beta).
       Symmetric PSD convention.
    """
    nv = V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
    L1 = np.linalg.norm(v1 - v2, axis=1)  # edge opposite vertex 0
    L2 = np.linalg.norm(v0 - v2, axis=1)  # edge opposite vertex 1
    L3 = np.linalg.norm(v0 - v1, axis=1)  # edge opposite vertex 2

    # Law-of-cosines angle at each vertex, same as cotLaplacian.m
    A1 = np.arccos(np.clip((L2**2 + L3**2 - L1**2) / (2 * L2 * L3), -1, 1))
    A2 = np.arccos(np.clip((L1**2 + L3**2 - L2**2) / (2 * L1 * L3), -1, 1))
    A3 = np.arccos(np.clip((L1**2 + L2**2 - L3**2) / (2 * L1 * L2), -1, 1))

    # Mirrors MATLAB's In/Jn/Sn assembly literally
    I = np.concatenate([i0, i1, i2])
    J = np.concatenate([i1, i2, i0])
    # S = 0.5 * cot([A3; A1; A2])
    S = 0.5 / np.tan(np.concatenate([A3, A1, A2]))

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])

    W = csr_matrix((Sn, (In, Jn)), shape=(nv, nv))
    return W


def analytic_geodesic_sphere(V: np.ndarray, src: int) -> np.ndarray:
    """Great-circle distance on a unit sphere from vertex `src`."""
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6), "V must be on unit sphere"
    dots = np.clip(V @ V[src], -1.0, 1.0)
    return np.arccos(dots)


def main():
    mesh = trimesh.creation.icosphere(subdivisions=3)  # 642 verts, 1280 triangles
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12)

    off_path = HERE / "icosphere_sub3.off"
    write_off(off_path, V, F)
    print(f"Wrote {off_path.name}  V={len(V)}  F={len(F)}")

    # --- Ground truth geodesic ---
    src = 0
    d_ref = analytic_geodesic_sphere(V, src)
    np.savetxt(HERE / "reference_analytic_geodesic.csv", d_ref, delimiter=",", fmt="%.17g")
    print(f"Wrote reference_analytic_geodesic.csv  min={d_ref.min():.6f}  max={d_ref.max():.6f}")
    print(f"  Antipode (argmax of analytic d) from vertex 0: vertex {int(np.argmax(d_ref))}, d = {d_ref.max():.6f}")
    print(f"  Expected max ~= pi = {np.pi:.6f}")

    # --- Reference cotangent Laplacian ---
    W = cot_laplacian(V, F)
    W_dense = W.toarray()
    np.savez_compressed(HERE / "reference_cot_laplacian.npz",
                        data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape))
    # A small dense CSV is handy for a visual diff against MATLAB (642x642 is ~3 MB)
    np.savetxt(HERE / "reference_cot_laplacian_dense.csv", W_dense, delimiter=",", fmt="%.10g")
    print(f"Wrote reference_cot_laplacian.{{npz,dense.csv}}")

    # --- Self-consistency checks on the Python reference ---
    checks = []
    def check(name: str, cond: bool, detail: str = ""):
        status = "PASS" if cond else "FAIL"
        checks.append((status, name, detail))
        print(f"  [{status}] {name}  {detail}")

    # Symmetry
    sym_err = np.abs(W_dense - W_dense.T).max()
    check("Laplacian is symmetric", sym_err < 1e-10, f"max|W-W^T|={sym_err:.2e}")

    # PSD (positive semi-definite): all eigvals >= -epsilon
    # For 642x642 use eigvalsh
    from numpy.linalg import eigvalsh
    evals = eigvalsh(W_dense)
    check("Laplacian is PSD", evals.min() > -1e-8, f"min eigval={evals.min():.2e}")

    # Row-sum near zero (Laplacian of constant = 0)
    row_sum_max = np.abs(W_dense.sum(axis=1)).max()
    check("Rows sum to zero (L*1 = 0)", row_sum_max < 1e-8, f"max|row sum|={row_sum_max:.2e}")

    # Kernel dim 1 (connected mesh => nullspace is constants only)
    near_zero = np.sum(evals < 1e-8)
    check("Kernel dim == 1 (connected)", near_zero == 1, f"eigvals<1e-8: {near_zero}")

    # Analytic geodesic invariants
    check("Analytic d[src] == 0", d_ref[src] == 0.0, f"d[{src}]={d_ref[src]}")
    check("Analytic d >= 0 everywhere", d_ref.min() >= 0.0, f"min d={d_ref.min()}")
    check("Analytic d max ~= pi (antipode on unit sphere)",
          abs(d_ref.max() - np.pi) < 1e-6,
          f"max d={d_ref.max():.6f} vs pi={np.pi:.6f}")

    with (HERE / "reference_invariants.txt").open("w") as f:
        for status, name, detail in checks:
            f.write(f"[{status}] {name}  {detail}\n")

    fails = [c for c in checks if c[0] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} FAIL(s). The Python reference itself is broken; fix before comparing to MATLAB.")
        raise SystemExit(1)

    # --- Write the "what to check once MATLAB runs" note ---
    note = HERE / "EXPECTED_MATLAB_OUTPUT.md"
    note.write_text(EXPECTED_MD.format(
        nv=len(V), nf=len(F), src_matlab=src + 1,
        argmax_analytic_matlab=int(np.argmax(d_ref)) + 1,
        max_d=d_ref.max(),
    ), encoding="utf-8")
    print(f"\nAll Python-side invariants pass. See {note.name} for the MATLAB-side checks.")


EXPECTED_MD = """# Validation plan for RGD_MATLAB on icosphere_sub3.off

## Inputs produced by validate_rgd_toy.py
- `icosphere_sub3.off` — unit sphere mesh, {nv} vertices, {nf} triangles (0-indexed on disk per the OFF spec; `readOff.m` adds +1 after parsing)
- `reference_analytic_geodesic.csv` — exact great-circle distance from vertex 0 (MATLAB index 1)
- `reference_cot_laplacian_dense.csv` — Python cotangent Laplacian in the same sign convention as `cotLaplacian.m` (positive diagonal, negative off-diagonals, row-sums zero)

## Tier 1 — unit level (cotLaplacian.m)
Run a one-off MATLAB snippet:
```matlab
Mm = MeshClass('icosphere_sub3');   % drops the .off; MeshClass prepends it
[W_mat, ~] = cotLaplacian(Mm);
writematrix(full(W_mat), 'matlab_cot_laplacian_dense.csv');
```
Then diff against `reference_cot_laplacian_dense.csv`. Acceptance:
- `max |W_py - W_mat|  < 1e-10`  (Frobenius err / nnz ~= machine epsilon)

If this fails, something is wrong at the matrix-assembly level and nothing else can be trusted.

## Tier 2 — toy-mesh RGD (demo.m-style)
Edit a copy of `demo.m` to point at `icosphere_sub3` with source `x0 = {src_matlab}`, sweep `alpha_hat0` across e.g. `[1e-3, 1e-2, 0.05, 0.25, 1.0]`, and for each run save `u_D1.csv`.

### Checks per alpha_hat
Numerical (no closed form at finite α̂):
- `u[src={src_matlab}] ~= 0`  (tolerance scales with alpha_hat)
- `u >= 0` everywhere
- `u` is symmetric under source swap: if we also run with source = argmax, u at vertex {src_matlab} should equal the original u at argmax (RGD is symmetric)

Convergence to analytic geodesic (as alpha_hat -> 0):
- As alpha_hat decreases, MATLAB `u` should approach `reference_analytic_geodesic.csv` up to an overall scale factor (RGD normalization may differ — check rank-correlation and per-vertex ratio stability rather than raw equality).
- Relative L2 between MATLAB `u/max(u)` and `d_ref/max(d_ref)` should decrease monotonically as alpha_hat -> 0.

### Argmax check
At reasonable alpha_hat (0.05-0.25), `argmax(u)` should be near the antipode of vertex {src_matlab}.
- Analytic argmax (MATLAB index) = {argmax_analytic_matlab}
- Tolerance: MATLAB's argmax should be that vertex OR a direct neighbor of it (subdivided icospheres don't always have an exact antipode; expected geodesic to the computed antipode ~= {max_d:.6f} rad).

## Tier 3 — gold standard (deferred)
Run `demo.m` on the actual CIELAB neural-bounded mesh (`external/matlab_rgd/RGB2CIELAB_neural_1.off`) with `alpha_hat = 1.25`, save `u_D1.csv` and `max_indices_*.txt`. Treat as reference for everything downstream. No further port validation needed unless we port.
"""


if __name__ == "__main__":
    main()
