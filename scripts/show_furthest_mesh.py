"""Visualize the CIELAB-space boundary mesh with each vertex colored by its
`furthest` output RGB. (Was `Mode.SHOW_MESH` in the old main.py.)
"""
import numpy as np

from arlabelvis.meshing import pointsToMesh, assign_vertex_colors

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD", alpha="75")


def main():
    cfg = CONFIG
    _, allLABs, _ = cfg.get_points()
    furthest = np.loadtxt(cfg.furthest_save_path, dtype=int)
    mesh = pointsToMesh(allLABs)
    mesh.visual.vertex_colors = assign_vertex_colors(mesh, allLABs, furthest)
    mesh.show()


if __name__ == "__main__":
    main()
