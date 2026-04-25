"""Short-paper tab:scene — per-frame label-color gradient statistics on the
dubai_changing_bgcolors scene, computed by running the scene processor with
three LUT constructions and reading the rendered-color-gradient column:

  1. Original — Kwon ΔE₀₀ baseline LUT.
  2. Smoothed (baseline) — 3D Gaussian filter applied directly to (1). σ=3.0
     matches the "naive" baseline discussion in the paper's §Method.
  3. Ours — CIELAB + neural-bounded + regularised geodesic, α̂=1.25.

Writes the final table to SHORTPAPER_TAB_DIR/scene/scene.tex under the same
LaTeX body paper.tex already \\input{}s.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from arlabelvis.scene_catalog import load_scene
from arlabelvis.scene_video import process_scene_video
from arlabelvis.metrics import load_and_filter, compute_gradients

from scripts.paper._configs import (LUT_CIELAB_DELTAE00,
                                    LUT_CIELAB_NEURAL_RGD_125)
from arlabelvis.luts import get_lut
from arlabelvis.luts import lut_to_srgb_u8
from scripts.paper._paths import tab_path


NAIVE_SIGMA = 3.0           # matches the "naive 3D Gaussian blur" baseline
SCENE_NAME = "dubai_changing_bgcolors"


def _naive_smoothed_lut(lut_u8: np.ndarray, sigma: float) -> np.ndarray:
    """3D Gaussian blur a (256,256,256,3) uint8 LUT per channel."""
    out = np.empty_like(lut_u8)
    for c in range(3):
        out[..., c] = gaussian_filter(lut_u8[..., c], sigma=sigma,
                                      mode="nearest")
    return out


def _scene_gradient_stats(lut_u8: np.ndarray, scene, tmp_dir: Path, tag: str
                          ) -> tuple[float, float, float, float]:
    """Run the scene processor with ``lut_u8`` and return
    (max, mean, median, std) of the per-frame rendered-color gradient."""
    csv_out = tmp_dir / f"{SCENE_NAME}__{tag}.csv"
    print(f"[tab_scene] {tag}: running processor -> {csv_out.name}")
    process_scene_video(str(scene.video), lut_u8, str(scene.label_mask),
                        str(csv_out), cec_source="label")
    df = load_and_filter(csv_out)
    grads = compute_gradients(df)
    rendered = grads["rendered"]["magnitude"].dropna()
    if len(rendered) == 0:
        raise RuntimeError(f"{tag}: no rendered gradient rows")
    return (float(rendered.max()), float(rendered.mean()),
            float(rendered.median()), float(rendered.std()))


def main():
    scene = load_scene(SCENE_NAME)
    assert scene.label_mask is not None, f"{SCENE_NAME} has no label mask"

    original_u8 = lut_to_srgb_u8(get_lut(LUT_CIELAB_DELTAE00),
                                 LUT_CIELAB_DELTAE00.output_space)
    smoothed_u8 = _naive_smoothed_lut(original_u8, sigma=NAIVE_SIGMA)
    ours_u8 = lut_to_srgb_u8(get_lut(LUT_CIELAB_NEURAL_RGD_125),
                             LUT_CIELAB_NEURAL_RGD_125.output_space)

    with tempfile.TemporaryDirectory(prefix="tab_scene_") as td:
        tmp_dir = Path(td)
        orig_stats = _scene_gradient_stats(original_u8, scene, tmp_dir, "original")
        naive_stats = _scene_gradient_stats(smoothed_u8, scene, tmp_dir, "smoothed")
        ours_stats = _scene_gradient_stats(ours_u8, scene, tmp_dir, "ours")

    # Column ordering: max, mean, median, std. Lower-is-better across all
    # four; bold the best per column.
    stats = [orig_stats, naive_stats, ours_stats]
    best = [int(np.argmin([s[c] for s in stats])) for c in range(4)]

    def cell(i_row, i_col, val, fmt="{:.2f}"):
        s = fmt.format(val)
        return r"\textbf{" + s + r"}" if best[i_col] == i_row else s

    rows = [
        (r"Original~\cite{KwonActive}",   orig_stats),
        (r"Smoothed (baseline)",          naive_stats),
        (r"Ours",                         ours_stats),
    ]

    lines = [
        r"\begin{table}[tb]",
        r"  \caption{Per-frame label-color gradient across a time-lapse video. "
        r"Lower is better across all four columns. The original"
        r"~\cite{KwonActive} concentrates mass at the median / mean end but "
        r"still exhibits large intermittent jumps (high \emph{max} and "
        r"standard deviation) that read as flicker.}",
        r"  \label{tab:scene}",
        r"  \scriptsize\centering",
        r"  \begin{tabular}{lrrrr}",
        r"  \toprule",
        r"  & \textbf{Grad.\ max} & \textbf{Grad.\ avg} & "
        r"\textbf{Grad.\ median} & \textbf{Grad.\ std.} \\",
        r"  \midrule",
    ]
    for i, (label, vals) in enumerate(rows):
        gmax, gavg, gmed, gstd = vals
        lines.append(
            f"  {label} & {cell(i, 0, gmax)} & {cell(i, 1, gavg)} "
            f"& {cell(i, 2, gmed)} & {cell(i, 3, gstd)} \\\\"
        )
    lines += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]

    out = tab_path("scene", "scene.tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
