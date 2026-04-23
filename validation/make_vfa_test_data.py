# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh"]
# ///
"""Generate a deterministic per-face vector field on icosphere_sub3 and write it
as a CSV for both MATLAB and Python to consume.

vf[f] = normalized (V[F[f, 1]] - V[F[f, 0]])  — an edge direction, tangent to
the face by construction.
"""
from pathlib import Path
import numpy as np
import trimesh

mesh = trimesh.creation.icosphere(subdivisions=3)
V = np.asarray(mesh.vertices)
F = np.asarray(mesh.faces, dtype=int)

edges = V[F[:, 1]] - V[F[:, 0]]
vf = edges / np.linalg.norm(edges, axis=1, keepdims=True)

out = Path(__file__).parent / "vfa_test_vf.csv"
np.savetxt(out, vf, delimiter=",", fmt="%.17g")
print(f"wrote {out}  shape={vf.shape}  max|norm-1|={np.abs(np.linalg.norm(vf, axis=1) - 1).max():.2e}")
