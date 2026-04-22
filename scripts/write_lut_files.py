"""Write the sparse candidate (LAB, RGB) LUT files from the `furthest` array produced
by run_rgd. These are the files the interpolate step upsamples to dense 256^3.
(Was `Mode.TO_FILE` in the old main.py.)
"""
import numpy as np

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD", alpha="75")


def main():
    cfg = CONFIG
    allRGBs, _, _ = cfg.get_points()
    furthest = np.loadtxt(cfg.furthest_save_path, dtype=int)

    with open(cfg.candidate_lab_file, "w") as fl, open(cfg.candidate_rgb_file, "w") as fr:
        for lab, rgb in zip(furthest.tolist(), allRGBs.tolist()):
            fl.write(f"{lab[0]},{lab[1]},{lab[2]}\n")
            fr.write(f"{rgb[0]},{rgb[1]},{rgb[2]}\n")
    print(f"saved {cfg.candidate_lab_file} + {cfg.candidate_rgb_file}")


if __name__ == "__main__":
    main()
