"""Smoke test for viz.lut_render using a synthetic LUT with no filesystem fixtures.

Generates a synthetic 32^3 sampled LUT (interpolated to 256^3 by nearest-neighbor),
writes it to a temp folder in the paired-text format, and runs all three renders.
Outputs land in viz/_smoke_out/ for visual inspection.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from arlabelvis.viz import (
    load_lut,
    render_srgb_cube_isometric,
    render_color_space_pointcloud,
    render_hue_histogram,
)

HERE = Path(__file__).parent
OUT = HERE / "_smoke_out"
OUT.mkdir(exist_ok=True)


def synth_lut_files(step: int) -> tuple[Path, Path, np.ndarray]:
    """A synthetic 'farthest color' LUT: assign each input RGB its simple complement.
    Store as AllCandidateLABvals_*.txt + AllCorrespondingRGBVals_*.txt with
    LAB values so the loader path exercises the lab2rgb conversion."""
    from skimage.color import rgb2lab

    grid = np.arange(0, 256, 256 // step, dtype=np.int32)  # step evenly-spaced
    R, G, B = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_in = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    # Assign complement as a trivial 'farthest color'
    rgb_out = 255 - rgb_in
    # Convert to LAB for storage (that's what main.py produces)
    lab_out = rgb2lab(rgb_out.astype(np.float32) / 255.0)

    lab_file = OUT / "synth_lab.txt"
    rgb_file = OUT / "synth_rgb.txt"
    np.savetxt(lab_file, lab_out, delimiter=",", fmt="%.6f")
    np.savetxt(rgb_file, rgb_in, delimiter=",", fmt="%d")

    # Points in CIELAB for the pointcloud render: convert the *input* RGB grid
    lab_in = rgb2lab(rgb_in.astype(np.float32) / 255.0)
    return lab_file, rgb_file, lab_in


def main():
    step = 16  # 16^3 = 4096 samples
    lab_file, rgb_file, points_lab = synth_lut_files(step)

    print(f"loading LUT from {lab_file.name} + {rgb_file.name}")
    lut = load_lut(lab_file, rgb_file, value="lab")
    print(f"  LUT shape: {lut.shape}, dtype: {lut.dtype}")
    print(f"  sample: lut[128,128,128] -> {lut[128, 128, 128].tolist()} (expect near complement of 128 = 127)")

    print("rendering sRGB cube...")
    render_srgb_cube_isometric(lut, save_path=OUT / "cube.png",
                              title="Synthetic LUT — sRGB cube isometric",
                              stride=4)

    print("rendering CIELAB point cloud...")
    render_color_space_pointcloud(
        lut, points_lab,
        save_path=OUT / "pointcloud_cielab.png",
        title="Synthetic LUT — CIELAB pointcloud",
        subsample=16,
    )

    print("rendering hue histogram...")
    render_hue_histogram(lut, save_path=OUT / "hue_hist.png",
                        title="Synthetic LUT — hue distribution")

    print(f"done. outputs in {OUT}")


if __name__ == "__main__":
    main()
