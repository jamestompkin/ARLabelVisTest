"""Figure: sRGB Euclidean farthest-color LUT as an isometric cube.

Thesis Figure label: fig:srgb_euclidean. Kwon-style illustration of the 8-color
collapse you get from Euclidean farthest-color on the sRGB cube — each input
picks the opposite cube corner, so the whole LUT takes only 8 output values.
The pathology that motivates the rest of the method.
"""
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_SRGB_EUCLIDEAN
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "srgb_euclidean.png"


def main():
    lut = get_lut(LUT_SRGB_EUCLIDEAN)
    render_srgb_cube_isometric(lut_to_srgb_u8(lut, LUT_SRGB_EUCLIDEAN), save_path=OUT,
                              title="sRGB Euclidean farthest-color LUT (cube corners)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
