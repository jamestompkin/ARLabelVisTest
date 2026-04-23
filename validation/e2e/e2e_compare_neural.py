"""E2E comparison on the neural-bounded CIELAB mesh.

Python vs MATLAB full pipeline:
  smooth_mesh (neural)  ->  all-pairs RGD  ->  furthest_rgd + interpolate  ->  render

Both backends consume the same `e2e_cielab_neural.off` (output of
`validation/e2e/build_e2e_neural_mesh.py`) and `matlab_max_indices_neural.csv`
(output of `run_matlab_e2e_neural.m`).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from skimage.color import lab2rgb

from arlabelvis.distances import furthest_rgd
from arlabelvis.interpolate import interpolate_interval
from arlabelvis.meshing import generate_LABs
from arlabelvis.off import read_off
from arlabelvis.rgd.allpairs import compute_all_pairs_argmax
from arlabelvis.viz import render_rgb_cube_isometric

INTERVAL = 16
ALPHA_HAT = 0.25
OFF_PATH = ROOT / "external/matlab_rgd/e2e_cielab_neural.off"
MATLAB_IDX = HERE / "matlab_max_indices_neural.csv"
OUT_DIR = HERE / "out_neural"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    V, F = read_off(str(OFF_PATH))
    print(f"mesh: {len(V)} verts, {len(F)} faces")

    print("\n[1/4] Python all-pairs RGD...")
    t0 = time.perf_counter()
    py_idx = compute_all_pairs_argmax(
        V, F, ALPHA_HAT, one_indexed=True, n_workers=10, progress_every=200
    )
    print(f"      {time.perf_counter()-t0:.1f}s total")

    mat_idx = np.loadtxt(MATLAB_IDX, delimiter=",", dtype=int).ravel()
    agree = int((mat_idx == py_idx).sum())
    print(f"\n[2/4] Argmax parity: {agree}/{len(py_idx)}  ({100*agree/len(py_idx):.2f}%)")

    py_idx_file = OUT_DIR / "python_max_indices_neural.csv"
    np.savetxt(py_idx_file, py_idx.reshape(1, -1), fmt="%d", delimiter=",")

    print("\n[3/4] Assembling final 256^3 LUTs via furthest_rgd + interpolate_interval...")
    allRGBs, allLABs = generate_LABs(stepSize=INTERVAL)

    def build(idx_file, tag):
        lut_path = OUT_DIR / f"{tag}_lut.npy"
        f = furthest_rgd(V, allLABs, allRGBs, str(idx_file))
        interpolate_interval(allRGBs, f, str(lut_path), INTERVAL)
        return lut_path

    py_lut_path = build(py_idx_file, "python")
    mat_lut_path = build(MATLAB_IDX, "matlab")

    py_lab = np.load(py_lut_path).astype(np.float32)
    mat_lab = np.load(mat_lut_path).astype(np.float32)

    lab_max = float(np.abs(py_lab - mat_lab).max())
    py_rgb = np.clip(lab2rgb(py_lab.reshape(-1, 3)).reshape(256, 256, 256, 3) * 255, 0, 255).astype(np.uint8)
    mat_rgb = np.clip(lab2rgb(mat_lab.reshape(-1, 3)).reshape(256, 256, 256, 3) * 255, 0, 255).astype(np.uint8)
    rgb_diff = np.abs(py_rgb.astype(np.int32) - mat_rgb.astype(np.int32))

    def grads(lut):
        gx, gy, gz = np.gradient(lut.astype(np.float32), axis=(0, 1, 2))
        mag = np.sqrt(gx**2 + gy**2 + gz**2).sum(axis=-1)
        return float(mag.max()), float(mag.mean()), float(mag.std())

    py_g, mat_g = grads(py_lab), grads(mat_lab)

    print(f"\n[4/4] DIFF:")
    print(f"  per-voxel LAB diff max: {lab_max:.3f}")
    print(f"  per-voxel sRGB diff:    max={rgb_diff.max()}/255  "
          f"p95={np.percentile(rgb_diff, 95):.2f}  mean={rgb_diff.mean():.4f}")
    print(f"  gradient stats (max, avg, std):")
    print(f"     Python: {py_g}")
    print(f"     MATLAB: {mat_g}")

    render_rgb_cube_isometric(py_rgb, save_path=OUT_DIR / "cube_python.png",
                              title="Python, neural-bounded CIELAB")
    render_rgb_cube_isometric(mat_rgb, save_path=OUT_DIR / "cube_matlab.png",
                              title="MATLAB, neural-bounded CIELAB")

    passed = (rgb_diff.max() == 0) or (np.percentile(rgb_diff, 95) < 3.0)
    print(f"\n{'PASS' if passed else 'FAIL'} (bar: p95 RGB diff < 3/255)")
    print(f"outputs in {OUT_DIR}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
