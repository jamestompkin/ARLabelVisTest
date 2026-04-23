"""Short-paper fig:hue_hist: three hue histograms stacked by LaTeX.

Each output is a single-strip PNG of the LUT's output-hue distribution.
Short paper compares the Kwon ΔE₀₀ baseline, CIELAB Euclidean, and our
neural-bounded RGD construction.

Outputs into SHORTPAPER_FIG_DIR:
  - OriginalLABVals_hue.png                                     Kwon baseline
  - AllCandidateLABvals_CIELAB_1_Euclidean_hue.png              CIELAB Euclidean
  - AllCandidateLABvals_CIELAB_1_RGD_50_neural_256_hue.png      neural + RGD α̂=0.5
"""
from arlabelvis.viz import render_hue_histogram

from scripts.paper._configs import (LUT_CIELAB_DELTAE00, LUT_CIELAB_EUCLIDEAN,
                                    LUT_CIELAB_NEURAL_RGD_050)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import lut_to_srgb_u8
from scripts.paper._shortpaper import shortpaper_path


def main():
    for cfg, filename, title in [
        (LUT_CIELAB_DELTAE00,
         "OriginalLABVals_hue.png",
         r"CIELAB $\Delta E_{00}$ (Kwon baseline)"),
        (LUT_CIELAB_EUCLIDEAN,
         "AllCandidateLABvals_CIELAB_1_Euclidean_hue.png",
         "CIELAB + Euclidean"),
        (LUT_CIELAB_NEURAL_RGD_050,
         "AllCandidateLABvals_CIELAB_1_RGD_50_neural_256_hue.png",
         r"Ours: CIELAB + neural + RGD ($\hat\alpha=0.5$)"),
    ]:
        lut = lut_to_srgb_u8(get_lut(cfg), cfg)
        render_hue_histogram(lut,
                             save_path=shortpaper_path("hue_histograms", filename),
                             title=f"Hue distribution — {title}")
        print(f"wrote hue_histograms/{filename}")


if __name__ == "__main__":
    main()
