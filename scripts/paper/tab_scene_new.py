"""tab_scene equivalent under the NEW algorithm: loads the pre-baked NEW
dense LUT for ``LUT_CIELAB_NEURAL_RGD_125`` and runs the same per-frame
gradient evaluation on dubai_changing_bgcolors. Reports OLD vs NEW for the
"Ours" row; the Original (Kwon) and Smoothed (Gaussian-on-LUT) rows are
unchanged because they use the bare-hull config (where OLD == NEW).

Reads:
  - ``results/shortpaper/validation/neural_rgd_125_new_lut.npy``  (50 MB,
    produced by ``scripts.paper.bake_neural_rgd_new``).

Writes:
  - ``results/shortpaper/validation/scene_old_vs_new.md``  (markdown report)
  - Does NOT overwrite ``tables/scene/scene.tex`` — production stays untouched
    until you decide to flip.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from arlabelvis.luts import get_lut, lut_to_srgb_u8
from arlabelvis.metrics import compute_gradients, load_and_filter
from arlabelvis.scene_catalog import load_scene
from arlabelvis.scene_video import process_scene_video
from scripts.paper._configs import (
    LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_125,
)
from scripts.paper._paths import SHORTPAPER_DIR

NAIVE_SIGMA = 3.0
SCENE_NAME = "dubai_changing_bgcolors"


def _naive_smoothed_lut(lut_u8: np.ndarray, sigma: float) -> np.ndarray:
    out = np.empty_like(lut_u8)
    for c in range(3):
        out[..., c] = gaussian_filter(lut_u8[..., c], sigma=sigma,
                                       mode="nearest")
    return out


def _scene_gradient_stats(lut_u8: np.ndarray, scene, tmp_dir: Path, tag: str
                          ) -> tuple[float, float, float, float]:
    csv_out = tmp_dir / f"{SCENE_NAME}__{tag}.csv"
    print(f"[tab_scene_new] {tag}: running processor -> {csv_out.name}",
          flush=True)
    t0 = time.perf_counter()
    process_scene_video(str(scene.video), lut_u8, str(scene.label_mask),
                        str(csv_out), cec_source="label")
    print(f"[tab_scene_new] {tag}: scene processed "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)
    df = load_and_filter(csv_out)
    grads = compute_gradients(df)
    rendered = grads["rendered"]["magnitude"].dropna()
    if len(rendered) == 0:
        raise RuntimeError(f"{tag}: no rendered gradient rows")
    return (float(rendered.max()), float(rendered.mean()),
            float(rendered.median()), float(rendered.std()))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    val_dir = SHORTPAPER_DIR / "validation"
    new_lut_path = val_dir / "neural_rgd_125_new_lut.npy"
    if not new_lut_path.exists():
        raise FileNotFoundError(
            f"NEW LUT not found at {new_lut_path}. Run "
            f"scripts.paper.bake_neural_rgd_new first."
        )

    print(f"[tab_scene_new] loading scene '{SCENE_NAME}'...", flush=True)
    scene = load_scene(SCENE_NAME)
    assert scene.label_mask is not None, f"{SCENE_NAME} has no label mask"

    print("[tab_scene_new] loading LUTs...", flush=True)
    original_u8 = lut_to_srgb_u8(get_lut(LUT_CIELAB_DELTAE00),
                                  LUT_CIELAB_DELTAE00.output_space)
    smoothed_u8 = _naive_smoothed_lut(original_u8, sigma=NAIVE_SIGMA)
    ours_old_u8 = lut_to_srgb_u8(get_lut(LUT_CIELAB_NEURAL_RGD_125),
                                  LUT_CIELAB_NEURAL_RGD_125.output_space)
    ours_new_u8 = np.load(new_lut_path)
    assert ours_new_u8.shape == (256, 256, 256, 3), \
        f"unexpected NEW LUT shape: {ours_new_u8.shape}"
    print(f"  original: {original_u8.shape} {original_u8.dtype}", flush=True)
    print(f"  smoothed: {smoothed_u8.shape} {smoothed_u8.dtype}", flush=True)
    print(f"  ours (OLD): {ours_old_u8.shape} {ours_old_u8.dtype}", flush=True)
    print(f"  ours (NEW): {ours_new_u8.shape} {ours_new_u8.dtype}", flush=True)

    with tempfile.TemporaryDirectory(prefix="tab_scene_new_") as td:
        tmp_dir = Path(td)
        orig_stats = _scene_gradient_stats(original_u8, scene, tmp_dir, "original")
        naive_stats = _scene_gradient_stats(smoothed_u8, scene, tmp_dir, "smoothed")
        ours_old_stats = _scene_gradient_stats(ours_old_u8, scene, tmp_dir, "ours_old")
        ours_new_stats = _scene_gradient_stats(ours_new_u8, scene, tmp_dir, "ours_new")

    rows = [
        ("Original (Kwon DeltaE00)", orig_stats),
        ("Smoothed (3D Gaussian baseline)", naive_stats),
        ("Ours OLD (production code today)", ours_old_stats),
        ("Ours NEW (orthogonal-axis algorithm)", ours_new_stats),
    ]

    report = (
        "# Per-frame label-color gradient: OLD vs NEW for the 'ours' row\n\n"
        f"Scene: `{SCENE_NAME}`. Same processor, same Kwon baseline, same\n"
        "naive-Gaussian baseline. Only difference is which LUT goes into the\n"
        "'Ours' row -- production-code OLD bake-out vs NEW orthogonal-axis\n"
        "bake-out.\n\n"
        "| Method | Grad. max | Grad. avg | Grad. median | Grad. std |\n"
        "|---|---|---|---|---|\n"
    )
    for label, vals in rows:
        gmax, gavg, gmed, gstd = vals
        report += (
            f"| {label} | {gmax:.2f} | {gavg:.2f} | {gmed:.2f} | {gstd:.2f} |\n"
        )
    report += (
        "\n## Reading the comparison\n\n"
        "The paper headline (217.95 -> 41.60 max-gradient, 23.40 -> 6.79 std)\n"
        "compares the 'Original' row to the 'Ours' row. Under OLD, those\n"
        "numbers were what the paper reported. Under NEW, the 'Ours NEW' row\n"
        "is the corresponding measurement, and the headline shifts to\n"
        f"**{orig_stats[0]:.2f} -> {ours_new_stats[0]:.2f} max-gradient, "
        f"{orig_stats[3]:.2f} -> {ours_new_stats[3]:.2f} std**.\n"
    )

    out_path = val_dir / "scene_old_vs_new.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n=== REPORT ===\n", flush=True)
    print(report, flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
