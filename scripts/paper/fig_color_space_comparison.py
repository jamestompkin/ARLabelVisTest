"""Figure: OKLAB vs CIELAB (same distance, same smoothing). Thesis fig:color_space_comparison.

Composites two cubes side-by-side in one PNG.
"""
import numpy as np
from PIL import Image
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_OKLAB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "color_space_comparison.png"


def main():
    tmp = FIG_DIR / "_tmp_cs"
    tmp.mkdir(exist_ok=True)
    paths = []
    for cfg, tag in [(LUT_OKLAB_EUCLIDEAN, "oklab"),
                     (LUT_CIELAB_EUCLIDEAN, "cielab")]:
        p = tmp / f"{tag}.png"
        render_srgb_cube_isometric(lut_to_srgb_u8(get_lut(cfg), cfg), save_path=p,
                                  title=f"{tag.upper()} + Euclidean")
        paths.append(p)

    # Horizontal composite
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs)
    w_total = sum(im.width for im in imgs)
    out = Image.new("RGB", (w_total, h), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    out.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
