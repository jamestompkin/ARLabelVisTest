"""Build a boundary triangle mesh of the input color space at the configured sigma/vox
and visualize it interactively. (Was `Mode.MESH_FILE` in the old main.py.)"""
from arlabelvis.meshing import pointsToMesh
from arlabelvis.viz import show_original_mesh_pyvista

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", sigma=0.0, vox=256)


def main():
    _, _, points = CONFIG.get_points()
    mesh = pointsToMesh(points, sigma=CONFIG.sigma, vox=CONFIG.vox)
    show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)
    print(f"built mesh for space={CONFIG.space}, sigma={CONFIG.sigma}, vox={CONFIG.vox}")


if __name__ == "__main__":
    main()
