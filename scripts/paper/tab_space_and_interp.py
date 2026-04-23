"""Table: color-space × interpolation-space comparison.

All rows hold ``metric=Euclidean`` and ``smoothing=none`` constant; what
varies is the intermediate color space (sRGB / CIELAB / OKLAB) used for the
farthest-color argmax, and the output interpolation space (sRGB or CIELAB).

This table absorbs the old tab_color_space and makes the "which interp space"
axis first-class — the CIELAB-interp row is now an explicit comparison point
rather than hiding inside a DeltaE76-labelled row.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv

from arlabelvis.metrics import color_diversity
from scripts.paper._configs import (LUT_SRGB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN,
                                    LUT_CIELAB_DELTAE76, LUT_OKLAB_EUCLIDEAN)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import TAB_DIR, lut_to_srgb_u8

OUT = TAB_DIR / "space_and_interp.tex"


def _row(label, cfg):
    lut = lut_to_srgb_u8(get_lut(cfg), cfg)
    f = lut.astype(np.float32)
    gx, gy, gz = np.gradient(f, axis=(0, 1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum(axis=-1)
    hsv = rgb_to_hsv((f / 255.0).reshape(-1, 3))
    d = color_diversity(lut, nn_sample_size=5000)
    return (label, cfg.space, cfg.interp_space,
            float(mag.max()), float(mag.mean()),
            float(hsv[:, 1].mean()), float(hsv[:, 2].mean()),
            d["unique_colors"], d["palette_effective_size"],
            d["hue_entropy_bits"], d["mean_nn_delta_e"])


def main():
    rows = [
        _row("sRGB (cube corners)", LUT_SRGB_EUCLIDEAN),
        _row("CIELAB",              LUT_CIELAB_EUCLIDEAN),
        _row(r"CIELAB ($\Delta E_{76}$ = this row)", LUT_CIELAB_DELTAE76),
        _row("OKLAB",               LUT_OKLAB_EUCLIDEAN),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\small",
        r"\begin{tabular}{@{}lllcccccccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Space} & \textbf{Interp} & "
        r"\textbf{Grad Max} & \textbf{Grad Avg} & "
        r"\textbf{Sat Avg} & \textbf{Int Avg} & \textbf{\#Colors} & "
        r"\textbf{Eff. Palette} & \textbf{Hue H (bits)} & \textbf{Mean NN $\Delta E$}\\",
        r"\midrule",
    ]
    for r in rows:
        label, space, interp, *metrics = r
        lines.append(
            f"Euclidean & {label} & {interp} & "
            f"{metrics[0]:.2f} & {metrics[1]:.3f} & "
            f"{metrics[2]:.3f} & {metrics[3]:.3f} & {metrics[4]} & "
            f"{metrics[5]:.1f} & {metrics[6]:.2f} & {metrics[7]:.2f}\\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Color-space and interpolation-space comparison. All rows "
        r"fix the argmax metric (Euclidean) and the geometry (no smoothing); "
        r"\emph{Space} is the intermediate color space used for the "
        r"farthest-color search, and \emph{Interp} is where the output LUT "
        r"interpolates between chosen hull vertices. The top row is the "
        r"trivial sRGB baseline (the hull of the input 0-255 cube is its 8 "
        r"corners, so each input picks the opposite corner). "
        r"CIELAB + CIELAB-interp "
        r"is Kwon 2019's $\Delta E_{76}$ baseline.}",
        r"\label{tab:space_and_interp}", r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
