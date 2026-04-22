"""End-to-end LUT generation, post-MATLAB: read `max_indices_*.txt`, compute
furthest, interpolate to dense 256^3 LUT. (Was `Mode.FULL` in the old main.py.)
"""
from arlabelvis.distances import (
    furthest_rgd, furthest_euclidean_lab_points,
)
from arlabelvis.interpolate import interpolate_interval
from arlabelvis.off import read_off

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD", alpha="75",
                   smoothing_mode="neural", voxel_dim=256)


def main():
    cfg = CONFIG
    allRGBs, _, points = cfg.get_points()

    if cfg.space == "RGB":
        vertices = allRGBs
    else:
        vertices, _ = read_off(cfg.original_off_path)

    if cfg.distance_measure == "RGD":
        furthest = furthest_rgd(vertices, points, allRGBs, cfg.matlab_indices_path)
    else:
        furthest = furthest_euclidean_lab_points(points)

    interpolate_interval(allRGBs, furthest, cfg.final_lab_path, cfg.interval)
    print(f"saved final LUT -> {cfg.final_lab_path}")


if __name__ == "__main__":
    main()
