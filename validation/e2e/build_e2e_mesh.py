# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Build the mesh both backends will run on.

A convex hull of the CIELAB point cloud at interval=16 — ~300 hull vertices,
small enough that MATLAB all-pairs finishes in under a minute, realistic as a
stand-in for the bounded CIELAB gamut mesh the thesis pipeline uses.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from arlabelvis.meshing import generate_LABs

INTERVAL = 16
OFF_PATH = ROOT / "external" / "matlab_rgd" / "e2e_cielab_hull.off"


def main():
    allRGBs, allLABs = generate_LABs(stepSize=INTERVAL)
    print(f"input grid: {len(allRGBs)} RGBs at stepSize={INTERVAL}")
    hull = ConvexHull(allLABs)
    V = allLABs[hull.vertices]                 # (nv_hull, 3)
    # Remap face indices from the original allLABs indices to 0..nv_hull-1
    remap = -np.ones(len(allLABs), dtype=np.int64)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    F = remap[hull.simplices]
    assert (F >= 0).all()
    print(f"hull mesh: {len(V)} verts, {len(F)} faces")

    with OFF_PATH.open("w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in F:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
    print(f"wrote {OFF_PATH}")


if __name__ == "__main__":
    main()
