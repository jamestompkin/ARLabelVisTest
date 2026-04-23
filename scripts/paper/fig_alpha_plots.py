"""Figure: 4-panel α̂ sweep plots (gradient max/avg, saturation/intensity avg).

Thesis Figure label: fig:alpha_plots. Recomputes the metrics from each LUT
in the α̂ sweep so the plot reflects the current pipeline rather than the
hand-curated `data/rgd_analysis.csv`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv

from scripts.paper._configs import LUT_ALPHA_SWEEP
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "alpha_plots.png"


def _metrics(lut_u8):
    lut = lut_u8.astype(np.float32)
    gx, gy, gz = np.gradient(lut, axis=(0, 1, 2))
    mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).sum(axis=-1)
    grad_max = float(mag.max())
    grad_avg = float(mag.mean())
    rgb_unit = (lut / 255.0).reshape(-1, 3)
    hsv = rgb_to_hsv(rgb_unit)
    sat = float(hsv[:, 1].mean())
    val = float(hsv[:, 2].mean())
    return grad_max, grad_avg, sat, val


def main():
    alphas = [c.alpha_hat for c in LUT_ALPHA_SWEEP]
    grad_max, grad_avg, sat, val = [], [], [], []
    for cfg in LUT_ALPHA_SWEEP:
        gm, ga, s, v = _metrics(lut_to_srgb_u8(get_lut(cfg), cfg))
        grad_max.append(gm); grad_avg.append(ga); sat.append(s); val.append(v)
        print(f"  α̂={cfg.alpha_hat:>5.2f}  grad_max={gm:.2f}  grad_avg={ga:.2f}  "
              f"sat={s:.3f}  val={v:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ah = r"$\hat{\alpha}$"
    for ax, name, ys in [(axes[0, 0], "Gradient Max",      grad_max),
                         (axes[0, 1], "Gradient Avg",      grad_avg),
                         (axes[1, 0], "Saturation Avg",    sat),
                         (axes[1, 1], "Intensity Avg",     val)]:
        ax.plot(alphas, ys, marker="o", linewidth=2, markersize=5)
        ax.set_xlabel(ah); ax.set_ylabel(name); ax.set_title(name)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
