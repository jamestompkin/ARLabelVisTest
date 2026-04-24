"""Short-paper tab:timing — LUT-construction wall-clock time.

Rebuilds each of the three methods from scratch (invalidates the cache, then
calls ``get_lut(cfg)``, which runs the full generate_LABs -> mesh -> farthest
-> interpolate chain), timing each end-to-end:

  1. Original — CIELAB + ΔE₀₀ (Kwon baseline).
  2. Smoothed (baseline) — time(Original) + 3D Gaussian filter of the result.
  3. Ours — CIELAB + neural-bounded + regularised geodesic, α̂=1.25.

Writes ``tables/timing/timing.tex`` to the Overleaf tables tree.

Cost: ~5 min total at ``interval=16`` on this machine; interval=1 would be
hours. The paper compares to Kwon's reported 41 min on 4 CPUs; for that
apples-to-apples comparison this script should be re-run at interval=1 on a
machine with matching core count.
"""
from __future__ import annotations

import platform
import time
from multiprocessing import cpu_count

import numpy as np
from scipy.ndimage import gaussian_filter

from scripts.paper._configs import (LUT_CIELAB_DELTAE00,
                                    LUT_CIELAB_NEURAL_RGD_125)
from scripts.paper._lut_cache import CACHE_DIR, get_lut
from scripts.paper._shared import lut_to_srgb_u8
from scripts.paper._shortpaper import shortpaper_table_path


NAIVE_SIGMA = 3.0   # matches shortpaper_tab_scene


def _invalidate(cfg) -> None:
    """Delete a LUT's cache files so the next get_lut call rebuilds."""
    key = cfg.key()
    for p in CACHE_DIR.glob(f"{key}*"):
        p.unlink()


def _time_build(cfg) -> float:
    """Invalidate and rebuild a LUT from scratch; return wall-clock seconds."""
    _invalidate(cfg)
    t0 = time.perf_counter()
    get_lut(cfg)
    return time.perf_counter() - t0


def _time_naive_smoothed(base_cfg, sigma: float) -> float:
    """Time = (rebuild base) + (3D Gaussian filter of the base LUT)."""
    base_dt = _time_build(base_cfg)
    lut_u8 = lut_to_srgb_u8(get_lut(base_cfg), base_cfg)
    t0 = time.perf_counter()
    out = np.empty_like(lut_u8)
    for c in range(3):
        out[..., c] = gaussian_filter(lut_u8[..., c], sigma=sigma,
                                      mode="nearest")
    return base_dt + (time.perf_counter() - t0)


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    m = seconds / 60.0
    if m < 60:
        return f"{m:.1f} min"
    return f"{m / 60:.1f} h"


def main():
    interval = LUT_CIELAB_DELTAE00.interval  # all three share this
    assert LUT_CIELAB_NEURAL_RGD_125.interval == interval

    print(f"[tab_timing] host {platform.node()}, {cpu_count()} CPUs, "
          f"interval={interval}")

    print("[tab_timing] building Original (CIELAB + DeltaE00)...")
    original_s = _time_build(LUT_CIELAB_DELTAE00)
    print(f"    {original_s:.1f} s")

    print("[tab_timing] building Smoothed baseline (Original + Gaussian)...")
    smoothed_s = _time_naive_smoothed(LUT_CIELAB_DELTAE00, NAIVE_SIGMA)
    print(f"    {smoothed_s:.1f} s")

    print("[tab_timing] building Ours (CIELAB + neural + RGD alpha=1.25)...")
    ours_s = _time_build(LUT_CIELAB_NEURAL_RGD_125)
    print(f"    {ours_s:.1f} s")

    rows = [
        (r"Original~\cite{KwonActive}", original_s,
         r"$\Delta E_{00}$ pairwise over the $256^3$ input cube"),
        (r"Smoothed (baseline)",       smoothed_s,
         fr"Original + 3D Gaussian ($\sigma={NAIVE_SIGMA:g}$)"),
        (r"Ours",                      ours_s,
         r"neural bound + RGD all-pairs ($\hat\alpha=1.25$)"),
    ]

    lines = [
        r"\begin{table}[tb]",
        r"  \caption{LUT-construction wall-clock time on "
        rf"a {cpu_count()}-core workstation at $\mathrm{{interval}}={interval}$. "
        r"Kwon~\cite{KwonActive} report 41~min on 4~CPUs for their construction "
        r"at full $256^3$ resolution; a like-for-like comparison requires "
        r"rerunning this table with $\mathrm{interval}=1$.}",
        r"  \label{tab:timing}",
        r"  \scriptsize\centering",
        r"  \begin{tabular}{llr}",
        r"  \toprule",
        r"  & \textbf{Stages} & \textbf{Wall-clock} \\",
        r"  \midrule",
    ]
    for label, seconds, desc in rows:
        lines.append(f"  {label} & {desc} & {_fmt(seconds)} \\\\")
    lines += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]

    out = shortpaper_table_path("timing", "timing.tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
