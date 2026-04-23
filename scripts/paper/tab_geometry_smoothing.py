"""Table: geometry-smoothing comparison (thesis Table 4.2 `tab:geometry_smoothing_comparison`).

Rows: None (no smoothing), σ=0.25 / 2.0 / 4.0 Gaussian, NB (neural-bounded).
Cols: Gradient Max, Gradient Avg, Saturation Avg, Intensity Avg,
      + diversity metrics (unique_colors, palette_effective_size,
                            hue_entropy_bits, mean_nn_delta_e).

Writes a LaTeX-includable .tex fragment.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv

from arlabelvis.metrics import color_diversity
from scripts.paper._configs import (CIELAB, LUT_CIELAB_GAUSSIAN_SIGMAS,
                                    LUT_CIELAB_NEURAL_RGD_025)
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import TAB_DIR, lut_to_srgb_u8

OUT = TAB_DIR / "geometry_smoothing.tex"

ROWS = [
    ("Hull",       CIELAB(smoothing="convex_hull", metric="RGD", alpha_hat=0.05)),
    (r"$\sigma=0.25$", LUT_CIELAB_GAUSSIAN_SIGMAS[0]),
    (r"$\sigma=2.0$",  LUT_CIELAB_GAUSSIAN_SIGMAS[1]),
    (r"$\sigma=4.0$",  LUT_CIELAB_GAUSSIAN_SIGMAS[2]),
    ("NB",         LUT_CIELAB_NEURAL_RGD_025),
]


def smoothness(lut_u8):
    lut = lut_u8.astype(np.float32)
    gx, gy, gz = np.gradient(lut, axis=(0, 1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum(axis=-1)
    rgb_unit = (lut / 255.0).reshape(-1, 3)
    hsv = rgb_to_hsv(rgb_unit)
    return {
        "grad_max": float(mag.max()),
        "grad_avg": float(mag.mean()),
        "sat_avg": float(hsv[:, 1].mean()),
        "int_avg": float(hsv[:, 2].mean()),
    }


def main():
    rows = []
    for label, cfg in ROWS:
        lut = lut_to_srgb_u8(get_lut(cfg), cfg)
        m = smoothness(lut)
        d = color_diversity(lut, nn_sample_size=5000)
        rows.append((label, m, d))
        print(f"  {label:>14s}  grad_max={m['grad_max']:8.2f}  "
              f"unique={d['unique_colors']:>7d}  hue_H={d['hue_entropy_bits']:.2f}")

    # Emit a minimal LaTeX booktabs table
    cols = r"@{}lcccccccc@{}"
    header = (r"\textbf{Method} & \textbf{Grad Max} & \textbf{Grad Avg} & "
              r"\textbf{Sat Avg} & \textbf{Int Avg} & \textbf{\#Colors} & "
              r"\textbf{Eff. Palette} & \textbf{Hue H (bits)} & \textbf{Mean NN $\Delta E$}\\")
    lines = [
        r"\begin{table}[H]",
        r"\small",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for label, m, d in rows:
        lines.append(
            f"{label} & {m['grad_max']:.2f} & {m['grad_avg']:.3f} & "
            f"{m['sat_avg']:.3f} & {m['int_avg']:.3f} & "
            f"{d['unique_colors']} & {d['palette_effective_size']:.1f} & "
            f"{d['hue_entropy_bits']:.2f} & {d['mean_nn_delta_e']:.2f}\\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Geometry smoothing comparison on the CIELAB color space, "
        r"with regularized geodesic distances at $\hat{\alpha}=0.05$. "
        r"Hull = unsmoothed convex hull of the CIELAB-mapped sRGB cube; "
        r"NB = neural bounding.}",
        r"\label{tab:geometry_smoothing_comparison}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
