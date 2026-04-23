"""Short-paper teaser assets: Kwon baseline cube + our smoothed hero cube.

Emits into the Overleaf ieeevis2026/figures dir:
  - old_lookup.png         cube render of the CIELAB ΔE₀₀ LUT (Kwon baseline)
  - neural_bounding_05.png cube render of the hero LUT (CIELAB + neural RGD α̂=0.05)

The third teaser panel (``flicker2``) is a photo and stays in the thesis
figures directory; the short paper falls through to ../figures/ for it.
"""
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import (LUT_CIELAB_DELTAE00,
                                    LUT_CIELAB_NEURAL_RGD_005)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import lut_to_srgb_u8
from scripts.paper._shortpaper import shortpaper_path


def main():
    cfg = LUT_CIELAB_DELTAE00
    render_srgb_cube_isometric(
        lut_to_srgb_u8(get_lut(cfg), cfg),
        save_path=shortpaper_path("teaser", "old_lookup.png"),
        title=r"CIELAB $\Delta E_{00}$ (Kwon baseline)",
    )

    cfg = LUT_CIELAB_NEURAL_RGD_005
    render_srgb_cube_isometric(
        lut_to_srgb_u8(get_lut(cfg), cfg),
        save_path=shortpaper_path("teaser", "neural_bounding_05.png"),
        title=r"Neural-bounded + RGD ($\hat\alpha=0.05$)",
    )

    print("wrote teaser/old_lookup.png, teaser/neural_bounding_05.png to SHORTPAPER_FIG_DIR")


if __name__ == "__main__":
    main()
