# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "trimesh", "scipy", "pyvista", "open3d", "scikit-image"]
# ///
"""Build a neural-bounded CIELAB mesh for e2e testing.

Decimated to a size where MATLAB all-pairs finishes in a minute or two so the
full-pipeline A/B completes in a session. The undecimated mesh (~500K faces,
from a 256-voxel binvox + loop subdivision) is the thesis-fidelity output;
decimation is only for this validation run.
"""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from arlabelvis.meshing import generate_LABs
from arlabelvis.bounding import neural_bounded_mesh
from arlabelvis.off import save_off_file

INTERVAL = 4      # only for computing the reference allLABs extent (not LUT size)
VOXEL_DIM = 256
TARGET_FACES = 2000
OFF_PATH = ROOT / "external" / "matlab_rgd" / "e2e_cielab_neural.off"
BINVOX = ROOT / f"data/neural_bounding_CIELAB_{VOXEL_DIM}.binvox"


def main():
    _, allLABs = generate_LABs(stepSize=INTERVAL)
    print(f"reference allLABs: {len(allLABs)} points  "
          f"L in [{allLABs[:,0].min():.1f}, {allLABs[:,0].max():.1f}]  "
          f"a in [{allLABs[:,1].min():.1f}, {allLABs[:,1].max():.1f}]  "
          f"b in [{allLABs[:,2].min():.1f}, {allLABs[:,2].max():.1f}]")

    mesh = neural_bounded_mesh(str(BINVOX), VOXEL_DIM, allLABs,
                               subdivide=True, target_faces=TARGET_FACES)
    print(f"neural-bounded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}, euler={mesh.euler_number}")
    import trimesh as tm
    comps = tm.graph.connected_components(mesh.edges)
    print(f"  connected components = {len(comps)}")
    save_off_file(str(OFF_PATH), mesh)


if __name__ == "__main__":
    main()
