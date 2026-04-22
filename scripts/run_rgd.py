"""Given the MATLAB `max_indices_*.txt` produced by external/matlab_rgd/demo.m,
compute the furthest-color RGB for each input color and save as a text file.
(Was `Mode.RGD` in the old main.py.)
"""
import numpy as np

from arlabelvis.distances import furthest_rgd
from arlabelvis.off import read_off

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD",
                   alpha="75", smoothing_mode="neural", voxel_dim=256)


def main():
    cfg = CONFIG
    allRGBs, _, points = cfg.get_points()

    vertices, _ = read_off(cfg.original_off_path)
    furthest = furthest_rgd(vertices, points, allRGBs, cfg.matlab_indices_path)

    np.savetxt(cfg.furthest_save_path, furthest, fmt="%d")
    print(f"saved furthest RGB values -> {cfg.furthest_save_path}")


if __name__ == "__main__":
    main()
