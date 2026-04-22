"""Visualize a MATLAB-produced geodesic distance field on the source mesh.
Reads `external/matlab_rgd/u_D1.csv` (written by demo.m) and overlays it on the .off mesh.
(Was `Mode.TEST_MATLAB` in the old main.py.)
"""
import numpy as np

from arlabelvis.viz import plot_geodesic_field_pyvista
from arlabelvis.off import read_off

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB")


def main():
    cfg = CONFIG
    dist = np.loadtxt("external/matlab_rgd/u_D1.csv")
    vertices, faces = read_off(f"external/matlab_rgd/RGB2{cfg.space}_{cfg.interval}.off")
    furthest_idx = int(np.argmax(dist))
    plot_geodesic_field_pyvista(vertices, faces, dist, source_idx=0, furthest_idx=furthest_idx)


if __name__ == "__main__":
    main()
