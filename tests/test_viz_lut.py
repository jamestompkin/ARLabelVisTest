"""Smoke test for the viz renderers using a synthetic in-memory LUT.

Builds a 256^3 'farthest color' LUT where every input RGB maps to its
complement, then exercises the three renderers. Outputs land in
tests/_smoke_out/ for visual inspection.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from arlabelvis.viz import (
    render_srgb_cube_isometric,
    render_color_space_pointcloud,
    render_hue_histogram,
)

HERE = Path(__file__).parent
OUT = HERE / "_smoke_out"
OUT.mkdir(exist_ok=True)


def complement_lut() -> np.ndarray:
    """Trivial 'farthest color' LUT: each input RGB -> its complement."""
    r, g, b = np.meshgrid(np.arange(256), np.arange(256), np.arange(256),
                          indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return (255 - rgb).astype(np.uint8)


def main():
    print("building synthetic complement LUT (256^3 uint8)...")
    lut = complement_lut()
    print(f"  LUT shape: {lut.shape}, dtype: {lut.dtype}")
    print(f"  sample: lut[128,128,128] = {lut[128, 128, 128].tolist()} "
          f"(expect 127,127,127)")

    print("rendering sRGB cube...")
    render_srgb_cube_isometric(lut, save_path=OUT / "cube.png",
                               title="Synthetic complement LUT")

    print("rendering CIELAB point cloud...")
    from skimage.color import rgb2lab
    step = 16  # 16^3 points for the scatter
    grid = np.arange(0, 256, 256 // step, dtype=np.int32)
    R, G, B = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_in = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    points_lab = rgb2lab(rgb_in.astype(np.float32) / 255.0)
    render_color_space_pointcloud(
        lut, points_lab,
        save_path=OUT / "pointcloud_cielab.png",
        title="Synthetic complement LUT — CIELAB pointcloud",
        subsample=16,
    )

    print("rendering hue histogram...")
    render_hue_histogram(lut, save_path=OUT / "hue_hist.png",
                         title="Synthetic complement LUT — hue distribution")

    print(f"done. outputs in {OUT}")


if __name__ == "__main__":
    main()
