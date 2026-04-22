"""Interpolate the sparse candidate LUT to the full 256^3 lookup.
(Was `Mode.INTERPOLATE` in the old main.py.)
"""
from arlabelvis.interpolate import interpolate_from_files

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD", alpha="75",
                   smoothing_mode="neural", voxel_dim=256)


def main():
    cfg = CONFIG
    interpolate_from_files(cfg.candidate_rgb_file, cfg.candidate_lab_file, cfg.final_lab_path)
    print(f"saved interpolated LUT -> {cfg.final_lab_path}")


if __name__ == "__main__":
    main()
