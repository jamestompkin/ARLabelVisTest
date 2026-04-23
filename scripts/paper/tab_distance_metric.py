"""Table: distance-metric comparison (replaces the former tab_distance_measure).

All rows hold ``space=CIELAB`` and ``interp_space=CIELAB`` constant; the only
thing varying is which metric drives the farthest-color argmax. This isolates
the "which distance function to use" question from the "which color space"
and "which interpolation space" questions handled by tab_space_and_interp.

Includes the explicit ΔE₇₆ row (== Euclidean-in-CIELAB by definition, kept for
reader clarity because Kwon 2019 names it this way) alongside ΔE₉₄, ΔE₀₀, and
RGD on the unsmoothed convex-hull gamut mesh.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv

from arlabelvis.metrics import color_diversity
from scripts.paper._configs import (CIELAB, LUT_CIELAB_DELTAE76,
                                    LUT_CIELAB_DELTAE94, LUT_CIELAB_DELTAE00,
                                    LUT_CIELAB_HULL_RGD_025)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import TAB_DIR, lut_to_srgb_u8

OUT = TAB_DIR / "distance_metric.tex"

# RGD with CIELAB interp (matches the other rows' interp space for a fair
# comparison of argmax metrics alone).
LUT_CIELAB_HULL_RGD_025_LAB_INTERP = CIELAB(
    smoothing="convex_hull", metric="RGD", alpha_hat=0.25,
    interp_space="CIELAB",
)


def _row(label, cfg):
    lut = lut_to_srgb_u8(get_lut(cfg), cfg)
    f = lut.astype(np.float32)
    gx, gy, gz = np.gradient(f, axis=(0, 1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum(axis=-1)
    hsv = rgb_to_hsv((f / 255.0).reshape(-1, 3))
    d = color_diversity(lut, nn_sample_size=5000)
    return (label, float(mag.max()), float(mag.mean()),
            float(hsv[:, 1].mean()), float(hsv[:, 2].mean()),
            d["unique_colors"], d["palette_effective_size"],
            d["hue_entropy_bits"], d["mean_nn_delta_e"])


def main():
    rows = [
        _row(r"$\Delta E_{76}$ (= Euclidean)", LUT_CIELAB_DELTAE76),
        _row(r"$\Delta E_{94}$",               LUT_CIELAB_DELTAE94),
        _row(r"$\Delta E_{00}$",               LUT_CIELAB_DELTAE00),
        _row("RGD",                            LUT_CIELAB_HULL_RGD_025_LAB_INTERP),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\small",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Grad Max} & \textbf{Grad Avg} & \textbf{Sat Avg} & "
        r"\textbf{Int Avg} & \textbf{\#Colors} & \textbf{Eff. Palette} & "
        r"\textbf{Hue H (bits)} & \textbf{Mean NN $\Delta E$}\\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r[0]} & {r[1]:.2f} & {r[2]:.3f} & {r[3]:.3f} & {r[4]:.3f} & "
            f"{r[5]} & {r[6]:.1f} & {r[7]:.2f} & {r[8]:.2f}\\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Distance-metric comparison. All rows use CIELAB as the "
        r"intermediate color space and CIELAB-space output interpolation. "
        r"$\Delta E_{76}$ equals Euclidean distance in CIELAB by definition "
        r"(hence the redundant label). RGD = regularised geodesic distances "
        r"on the unsmoothed convex-hull gamut mesh ($\hat\alpha=0.25$).}",
        r"\label{tab:distance_metric}", r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
