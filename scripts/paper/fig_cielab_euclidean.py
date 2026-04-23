"""Figure: CIELAB Euclidean farthest-color LUT as an isometric cube.

Thesis Figure label: fig:cielab_euclidean.
"""
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_CIELAB_EUCLIDEAN
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "cielab_euclidean.png"


def main():
    lut = get_lut(LUT_CIELAB_EUCLIDEAN)
    render_srgb_cube_isometric(lut_to_srgb_u8(lut, LUT_CIELAB_EUCLIDEAN), save_path=OUT,
                              title="CIELAB Euclidean farthest-color LUT")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
