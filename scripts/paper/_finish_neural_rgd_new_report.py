"""Recovery: re-run stage 7 of bake_neural_rgd_new from saved artifacts.

The main bake crashed at the report-write step on a Unicode codepoint in the
Windows cp1252 console encoding. Both LUTs were saved successfully (dense
NEW LUT plus the cached OLD LUT). This script loads them, recomputes the
comparison metrics, and writes the report with ASCII only + explicit utf-8
encoding so this can't bite us again.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from arlabelvis.luts import LutConfig, get_lut, lut_to_srgb_u8
from scripts.paper._configs import LUT_CIELAB_NEURAL_RGD_125
from scripts.paper._paths import SHORTPAPER_DIR


def _max_gradient_and_std(lut_u8: np.ndarray) -> tuple[float, float]:
    f = lut_u8.astype(np.float32)
    gx, gy, gz = np.gradient(f, axis=(0, 1, 2))
    mag = np.sqrt(gx**2 + gy**2 + gz**2).sum(axis=-1)
    return float(mag.max()), float(mag.std())


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = SHORTPAPER_DIR / "validation"
    new_lut_path = out_dir / "neural_rgd_125_new_lut.npy"
    ckpt_path = out_dir / "neural_rgd_125_checkpoint.npz"

    print("[recover] loading NEW LUT + checkpoint metadata...", flush=True)
    new_lut = np.load(new_lut_path)
    print(f"  NEW LUT shape={new_lut.shape}, dtype={new_lut.dtype}", flush=True)

    ckpt = np.load(ckpt_path)
    n_verts = len(ckpt["verts"])
    n_faces = len(ckpt["faces"])
    n_admit = int(ckpt["admit"].sum())
    admitted_mesh_verts = np.unique(ckpt["voxel_to_vert"][ckpt["admit"]])
    n_admitted_verts = len(admitted_mesh_verts)
    n_voxels_input = len(ckpt["admit"])

    cfg_old = LutConfig(
        working_space=LUT_CIELAB_NEURAL_RGD_125.working_space,
        shape=LUT_CIELAB_NEURAL_RGD_125.shape,
        metric=LUT_CIELAB_NEURAL_RGD_125.metric,
        metric_rgd_alpha_hat=LUT_CIELAB_NEURAL_RGD_125.metric_rgd_alpha_hat,
    )
    print("[recover] loading OLD LUT...", flush=True)
    t0 = time.perf_counter()
    old_raw = get_lut(cfg_old)
    old_lut = lut_to_srgb_u8(old_raw, cfg_old.output_space)
    print(f"  OLD LUT shape={old_lut.shape}  ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    print("[recover] computing metrics...", flush=True)
    t0 = time.perf_counter()
    new_max, new_std = _max_gradient_and_std(new_lut)
    old_max, old_std = _max_gradient_and_std(old_lut)
    n_voxels = new_lut.size // 3
    diff_voxels = (new_lut.reshape(-1, 3) != old_lut.reshape(-1, 3)).any(axis=1).sum()
    rgb_diff = np.abs(new_lut.astype(np.int32) - old_lut.astype(np.int32))
    mean_rgb_diff = float(rgb_diff.sum(axis=-1).mean())
    max_rgb_diff = int(rgb_diff.sum(axis=-1).max())
    print(f"  metrics done  ({time.perf_counter()-t0:.1f}s)", flush=True)

    report_text = (
        "# LUT_CIELAB_NEURAL_RGD_125 -- OLD vs NEW algorithm\n\n"
        "Baked under both algorithms at the full 256^3 dense LUT resolution.\n"
        "NEW implements the orthogonal-axis framing: admit MLP-inside gamut\n"
        "samples, restrict argmax to admitted-mapped mesh verts, output is\n"
        "the closest admitted gamut sample whose mesh vert was the argmax.\n\n"
        "## Mesh and admission\n\n"
        f"- Vertices: {n_verts:,}\n"
        f"- Faces:    {n_faces:,}\n"
        f"- Admitted gamut voxels: {n_admit:,} / {n_voxels_input:,} "
        f"({100*n_admit/n_voxels_input:.1f}%)\n"
        f"- Mesh verts that received at least one admitted voxel: "
        f"{n_admitted_verts:,} / {n_verts:,}\n\n"
        "## Per-voxel comparison (NEW vs OLD on the dense 256^3 LUT)\n\n"
        f"- Voxels where output differs: **{diff_voxels:,} / {n_voxels:,}** "
        f"({100*diff_voxels/n_voxels:.2f}%)\n"
        f"- Mean per-voxel L1 sRGB diff: {mean_rgb_diff:.2f}\n"
        f"- Max per-voxel L1 sRGB diff:  {max_rgb_diff}\n\n"
        "## Gradient metrics on the dense LUT\n\n"
        "| Algorithm | max-gradient | std |\n"
        "|---|---|---|\n"
        f"| OLD | {old_max:.2f} | {old_std:.2f} |\n"
        f"| NEW | {new_max:.2f} | {new_std:.2f} |\n"
        f"| delta (NEW - OLD) | {new_max - old_max:+.2f} "
        f"| {new_std - old_std:+.2f} |\n\n"
        "## Paper headline (for reference)\n\n"
        "The thesis abstract reports max-gradient 217.95 -> 41.60 and std\n"
        "23.40 -> 6.79 for the same recipe (Kwon's DeltaE00 baseline ->\n"
        "ours: neural+RGD alpha-hat=1.25). Those numbers came from the OLD\n"
        f"bake-out. Under NEW the 'ours' entry would be ({new_max:.2f}, "
        f"{new_std:.2f}).\n\n"
        "## Files\n\n"
        "- `neural_rgd_125_new_lut.npy`     -- NEW dense LUT (256^3 x 3 uint8)\n"
        "- `neural_rgd_125_checkpoint.npz`  -- mesh + per-vert RGD argmax + admission\n"
        f"- OLD dense LUT lives in the LutCache under key `{cfg_old.key()}`\n"
    )
    report_path = out_dir / "neural_rgd_125_old_vs_new.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n=== REPORT ===\n", flush=True)
    print(report_text, flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
