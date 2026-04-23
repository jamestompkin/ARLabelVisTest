"""Figure: neural-bounded CIELAB LUT (thesis fig:neural_bounding_comparison)."""
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_CIELAB_NEURAL_RGD_025
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "neural_bounded_lut.png"


def main():
    cfg = LUT_CIELAB_NEURAL_RGD_025
    render_srgb_cube_isometric(lut_to_srgb_u8(get_lut(cfg), cfg),
                              save_path=OUT, title="CIELAB neural-bounded LUT (α̂=0.25)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
