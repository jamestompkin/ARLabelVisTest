"""Figure: hue histograms for Euclidean, DeltaE76 (≈ΔE₀₀), and RGD LUTs.

Thesis figures: fig:euclidean_hist, fig:original_hist, fig:final_histogram.
"""
from arlabelvis.viz import render_hue_histogram

from scripts.paper._configs import (LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_DELTAE76,
                                    LUT_CIELAB_NEURAL_RGD_025)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT_EUCLIDEAN = FIG_DIR / "hue_histogram_euclidean.png"
OUT_DELTAE76 = FIG_DIR / "hue_histogram_deltae76.png"
OUT_RGD = FIG_DIR / "hue_histogram_rgd.png"


def main():
    for cfg, out, label in [
        (LUT_CIELAB_EUCLIDEAN, OUT_EUCLIDEAN, "CIELAB + Euclidean"),
        (LUT_CIELAB_DELTAE76, OUT_DELTAE76, "CIELAB + ΔE76 (Kwon baseline)"),
        (LUT_CIELAB_NEURAL_RGD_025, OUT_RGD, "CIELAB + neural-bounded + RGD (α̂=0.25)"),
    ]:
        lut = lut_to_srgb_u8(get_lut(cfg), cfg)
        render_hue_histogram(lut, save_path=out, title=f"Hue distribution — {label}")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
