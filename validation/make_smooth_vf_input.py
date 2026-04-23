# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh"]
# ///
"""Build a sparse input vector field on icosphere_sub3 for smooth_vf validation.

Sets a non-zero tangent direction at 4 hand-picked faces (well-separated around
the sphere so the smoothing has something non-trivial to do), zero elsewhere.

Each constrained vector is projected into the face's tangent plane so it is a
valid input (smooth_vf treats its input as face-tangent).
"""
from pathlib import Path
import numpy as np
import trimesh

mesh = trimesh.creation.icosphere(subdivisions=3)
V = np.asarray(mesh.vertices, dtype=float)
F = np.asarray(mesh.faces, dtype=int)

v0 = V[F[:, 0]]
v1 = V[F[:, 1]]
v2 = V[F[:, 2]]
Nf = np.cross(v0 - v1, v0 - v2)
Nf = Nf / np.linalg.norm(Nf, axis=1, keepdims=True)

# A directional field in world space
world_dir = np.array([[1.0, 0.0, 0.0]])

# Pick 4 well-separated constraint faces by sorting along axes
centroids = (v0 + v1 + v2) / 3.0
idx_posz = int(np.argmax(centroids[:, 2]))
idx_negz = int(np.argmin(centroids[:, 2]))
idx_posy = int(np.argmax(centroids[:, 1]))
idx_negy = int(np.argmin(centroids[:, 1]))
constrained = [idx_posz, idx_negz, idx_posy, idx_negy]

vf = np.zeros_like(Nf)
for i in constrained:
    # Project +x onto this face's tangent plane so it is a valid input
    proj = world_dir[0] - np.dot(world_dir[0], Nf[i]) * Nf[i]
    n = np.linalg.norm(proj)
    if n > 1e-6:
        vf[i] = proj / n

out = Path(__file__).parent / "smooth_vf_input.csv"
np.savetxt(out, vf, delimiter=",", fmt="%.17g")
print(f"wrote {out}  shape={vf.shape}  constrained faces={constrained}")
