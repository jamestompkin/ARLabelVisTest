# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh"]
# ///
"""Build a non-symmetric, production-sized mesh for stress-testing all-pairs RGD.

Starts from a subdivided icosphere, then radially modulates each vertex by a
smooth spatial function of its coordinates. The result:

  - No antipodal symmetry (breaks the "unique antipode = unique argmax" regime
    we had on the pure icosphere).
  - No crossings (modulation is smooth and bounded well above zero).
  - 2562 vertices, 5120 faces — comparable to real CIELAB-mesh scale.

Writes `external/matlab_rgd/deformed_ico_sub4.off`, consumed by both the
MATLAB and Python all-pairs drivers.
"""
from pathlib import Path
import numpy as np
import trimesh


def main():
    # subdivisions=4 gives 2562 verts, 5120 faces — enough to be real, small
    # enough that MATLAB all-pairs runs in ~10 s.
    mesh = trimesh.creation.icosphere(subdivisions=4)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)

    # Radial modulation: each vertex v gets its radius scaled by r(v). r is a
    # smooth function of the vertex's direction; adjacent vertices have similar
    # r, so the mesh stays non-self-intersecting.  Range roughly [0.72, 1.28].
    x, y, z = V.T
    r = 1.0 + 0.15 * np.sin(3.0 * x) * np.cos(2.0 * y) + 0.10 * np.sin(1.5 * z + 0.7)
    V_def = V * r[:, None]

    # Sanity: check nothing collapsed or blew up
    radii = np.linalg.norm(V_def, axis=1)
    print(f"deformed radii: min={radii.min():.3f}  max={radii.max():.3f}  "
          f"mean={radii.mean():.3f}")

    out = Path("external/matlab_rgd/deformed_ico_sub4.off")
    with out.open("w") as f:
        f.write(f"OFF\n{len(V_def)} {len(F)} 0\n")
        for v in V_def:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in F:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    print(f"wrote {out}  ({len(V_def)} verts, {len(F)} faces)")


if __name__ == "__main__":
    main()
