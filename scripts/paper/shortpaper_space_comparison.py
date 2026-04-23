"""Short-paper fig:space_comparison: sRGB / OKLAB / CIELAB side-by-side.

Three individual cube renders; LaTeX does the composition. All three use
``no geometry smoothing'' (= the unsmoothed convex-hull gamut mesh for the
RGD rows) at ``\\hat\\alpha = 0.05``.

Outputs into SHORTPAPER_FIG_DIR:
  - rgb_euclidean.png    sRGB + Euclidean (degenerate 8-corner baseline)
  - oklab_rgd_05.png     OKLAB + hull + RGD (alpha_hat = 0.05)
  - cielab_rgd_05.png    CIELAB + hull + RGD (alpha_hat = 0.05)
"""
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import (LUT_SRGB_EUCLIDEAN,
                                    LUT_OKLAB_HULL_RGD_005,
                                    LUT_CIELAB_HULL_RGD_005)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import lut_to_srgb_u8
from scripts.paper._shortpaper import shortpaper_path


def main():
    for cfg, filename, title in [
        (LUT_SRGB_EUCLIDEAN,        "rgb_euclidean.png",  "sRGB + Euclidean"),
        (LUT_OKLAB_HULL_RGD_005,    "oklab_rgd_05.png",   r"OKLAB + RGD ($\hat\alpha=0.05$)"),
        (LUT_CIELAB_HULL_RGD_005,   "cielab_rgd_05.png",  r"CIELAB + RGD ($\hat\alpha=0.05$)"),
    ]:
        render_srgb_cube_isometric(
            lut_to_srgb_u8(get_lut(cfg), cfg),
            save_path=shortpaper_path("space_comparison", filename),
            title=title,
        )
        print(f"wrote space_comparison/{filename}")


if __name__ == "__main__":
    main()
