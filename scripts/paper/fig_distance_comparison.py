"""Figure: distance-measure comparison (Euclidean / DeltaE76 / RGD) on CIELAB.

Thesis fig:distance_comparison.
"""
from PIL import Image
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import (LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_DELTAE76,
                                    LUT_CIELAB_HULL_RGD_025)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "distance_comparison.png"


def main():
    tmp = FIG_DIR / "_tmp_dist"
    tmp.mkdir(exist_ok=True)
    panels = [
        (LUT_CIELAB_EUCLIDEAN, "euclidean", "Euclidean"),
        (LUT_CIELAB_DELTAE76, "deltae76", "ΔE₇₆ (≈ Kwon)"),
        (LUT_CIELAB_HULL_RGD_025, "rgd", "RGD α̂=0.25"),
    ]
    paths = []
    for cfg, tag, title in panels:
        p = tmp / f"{tag}.png"
        render_srgb_cube_isometric(lut_to_srgb_u8(get_lut(cfg), cfg), save_path=p, title=title)
        paths.append(p)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs); w_total = sum(im.width for im in imgs)
    out = Image.new("RGB", (w_total, h), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, 0)); x += im.width
    out.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
