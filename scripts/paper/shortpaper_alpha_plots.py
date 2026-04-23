"""Short-paper fig:alpha_plots. Reuses the thesis-style alpha_plots figure,
re-running the plotting step with the same LUT cache so the output lands in
SHORTPAPER_FIG_DIR instead of (or in addition to) results/paper/figures.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

from scripts.paper._configs import LUT_ALPHA_SWEEP
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import lut_to_srgb_u8
from scripts.paper._shortpaper import shortpaper_path


def _metrics(lut_u8):
    lut = lut_u8.astype(np.float32)
    gx, gy, gz = np.gradient(lut, axis=(0, 1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum(axis=-1)
    rgb_unit = (lut / 255.0).reshape(-1, 3)
    hsv = rgb_to_hsv(rgb_unit)
    return float(mag.max()), float(mag.mean()), float(hsv[:, 1].mean()), float(hsv[:, 2].mean())


def main():
    alphas = [c.alpha_hat for c in LUT_ALPHA_SWEEP]
    grad_max, grad_avg, sat, val = [], [], [], []
    for cfg in LUT_ALPHA_SWEEP:
        gm, ga, s, v = _metrics(lut_to_srgb_u8(get_lut(cfg), cfg))
        grad_max.append(gm); grad_avg.append(ga); sat.append(s); val.append(v)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ah = r"$\hat{\alpha}$"
    for ax, name, ys in [(axes[0, 0], "Gradient Max",   grad_max),
                         (axes[0, 1], "Gradient Avg",   grad_avg),
                         (axes[1, 0], "Saturation Avg", sat),
                         (axes[1, 1], "Intensity Avg",  val)]:
        ax.plot(alphas, ys, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel(ah); ax.set_ylabel(name); ax.set_title(name)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = shortpaper_path("alpha_plots", "alpha_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote alpha_plots/alpha_plots.png")


if __name__ == "__main__":
    main()
