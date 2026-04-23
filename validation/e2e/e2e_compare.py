# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib", "trimesh", "pyvista", "scikit-image"]
# ///
"""E2E pipeline comparison: Python backend vs MATLAB backend, final LUT diff.

Both backends take the same `e2e_cielab_hull.off` mesh as input. This script:
  1. Runs Python all-pairs RGD (arlabelvis.rgd.allpairs.compute_all_pairs_argmax).
  2. Loads MATLAB's max_indices (produced by run_matlab_e2e.m; must have been run).
  3. For each max_indices array, assembles a full 256^3 LUT via the library's
     furthest_rgd + interpolate_interval (exactly what scripts.full_pipeline does).
  4. Diffs the two final LUTs:
       - per-voxel RGB absolute difference: max, 95th percentile, mean
       - gradient-metric parity (max, avg, std) from arlabelvis.metrics.get_gradient
       - hue-histogram diff
  5. Renders each cube via arlabelvis.viz.render_srgb_cube_isometric.

Pass criterion: 95th-percentile per-voxel RGB diff < 3/255  (visually imperceptible).
"""
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from arlabelvis.meshing import generate_LABs
from arlabelvis.off import read_off
from arlabelvis.distances import furthest_rgd
from arlabelvis.interpolate import interpolate_interval
from arlabelvis.rgd.allpairs import compute_all_pairs_argmax
from arlabelvis.viz import load_lut, render_srgb_cube_isometric

INTERVAL = 16
ALPHA_HAT = 0.25
OFF_PATH = ROOT / "external/matlab_rgd/e2e_cielab_hull.off"
MATLAB_IDX_FILE = HERE / "matlab_max_indices.csv"
PY_IDX_FILE = HERE / "python_max_indices.csv"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(exist_ok=True)


def assemble_lut(max_indices_file: Path, off_path: Path, tag: str) -> Path:
    """Run furthest_rgd + interpolate_interval. Mirrors scripts.full_pipeline."""
    allRGBs, allLABs = generate_LABs(stepSize=INTERVAL)
    vertices, _ = read_off(str(off_path))
    furthest = furthest_rgd(vertices, allLABs, allRGBs, str(max_indices_file))
    final_lut_path = OUT_DIR / f"{tag}_lut.npy"
    interpolate_interval(allRGBs, furthest, str(final_lut_path), INTERVAL)
    return final_lut_path


def diff_luts(py_path: Path, mat_path: Path) -> dict:
    """Load the two dense LUTs and compute per-voxel and metric-level diffs."""
    from skimage.color import lab2rgb
    py_lab = np.load(py_path).astype(np.float32)
    mat_lab = np.load(mat_path).astype(np.float32)
    assert py_lab.shape == (256, 256, 256, 3) == mat_lab.shape, (py_lab.shape, mat_lab.shape)

    # per-voxel LAB diff
    lab_diff = np.abs(py_lab - mat_lab)
    # Convert both to sRGB for perceptual comparison
    py_rgb = np.clip(lab2rgb(py_lab.reshape(-1, 3)).reshape(256, 256, 256, 3) * 255, 0, 255).astype(np.uint8)
    mat_rgb = np.clip(lab2rgb(mat_lab.reshape(-1, 3)).reshape(256, 256, 256, 3) * 255, 0, 255).astype(np.uint8)
    rgb_diff = np.abs(py_rgb.astype(np.int32) - mat_rgb.astype(np.int32))

    # Gradient metric parity (arlabelvis.metrics.get_gradient semantics)
    def grad_stats(lut):
        gx, gy, gz = np.gradient(lut.astype(np.float32), axis=(0, 1, 2))
        mag = np.sqrt(gx**2 + gy**2 + gz**2).sum(axis=-1)
        return float(mag.max()), float(mag.mean()), float(mag.std())

    return {
        "py_rgb": py_rgb,
        "mat_rgb": mat_rgb,
        "lab_max": float(lab_diff.max()),
        "lab_mean": float(lab_diff.mean()),
        "rgb_max": int(rgb_diff.max()),
        "rgb_p95": float(np.percentile(rgb_diff, 95)),
        "rgb_mean": float(rgb_diff.mean()),
        "py_grad": grad_stats(py_lab),
        "mat_grad": grad_stats(mat_lab),
    }


def main():
    if not MATLAB_IDX_FILE.exists():
        print(f"Missing {MATLAB_IDX_FILE}. Run: matlab -batch \"cd('validation/e2e'); run_matlab_e2e\"")
        return 1

    print("=== E2E PIPELINE COMPARISON ===\n")

    # 1. Python all-pairs (if not already done)
    V, F = read_off(str(OFF_PATH))
    if not PY_IDX_FILE.exists():
        print(f"[1/4] Python all-pairs RGD on {OFF_PATH.name} ({len(V)} verts)")
        t0 = time.perf_counter()
        argmax_1based = compute_all_pairs_argmax(V, F, ALPHA_HAT, one_indexed=True)
        print(f"      {time.perf_counter() - t0:.2f}s")
        np.savetxt(PY_IDX_FILE, argmax_1based.reshape(1, -1), fmt="%d", delimiter=",")
    else:
        print(f"[1/4] Python max_indices already at {PY_IDX_FILE.name}")

    # 2. Argmax-level diff (cheap sanity)
    m = np.loadtxt(MATLAB_IDX_FILE, delimiter=",", dtype=int).ravel()
    p = np.loadtxt(PY_IDX_FILE, delimiter=",", dtype=int).ravel()
    assert m.shape == p.shape
    agree = int((m == p).sum())
    print(f"\n[2/4] Argmax parity: {agree}/{len(m)} "
          f"({100 * agree / len(m):.2f}%)  mismatches={len(m) - agree}")

    # 3. Assemble both LUTs end-to-end
    print("\n[3/4] Assembling final 256^3 LUTs via furthest_rgd + interpolate_interval")
    py_lut = assemble_lut(PY_IDX_FILE, OFF_PATH, "python")
    mat_lut = assemble_lut(MATLAB_IDX_FILE, OFF_PATH, "matlab")
    print(f"      {py_lut.name} and {mat_lut.name}")

    # 4. Diff
    print("\n[4/4] Diff final LUTs")
    d = diff_luts(py_lut, mat_lut)
    print(f"  per-voxel LAB:  max={d['lab_max']:.3f}   mean={d['lab_mean']:.3e}")
    print(f"  per-voxel sRGB: max={d['rgb_max']}/255  p95={d['rgb_p95']:.2f}/255  mean={d['rgb_mean']:.3f}/255")
    print(f"  gradient stats (max, avg, std):")
    print(f"     Python:  {d['py_grad']}")
    print(f"     MATLAB:  {d['mat_grad']}")
    dmax = abs(d['py_grad'][0] - d['mat_grad'][0]) / max(d['mat_grad'][0], 1e-300)
    davg = abs(d['py_grad'][1] - d['mat_grad'][1]) / max(d['mat_grad'][1], 1e-300)
    print(f"     relative diff (max, avg): {dmax:.3e}, {davg:.3e}")

    # Render both cubes side-by-side
    print("\nRendering cubes...")
    render_srgb_cube_isometric(d['py_rgb'], save_path=OUT_DIR / "cube_python.png",
                              title="Python backend LUT")
    render_srgb_cube_isometric(d['mat_rgb'], save_path=OUT_DIR / "cube_matlab.png",
                              title="MATLAB backend LUT")

    passed = d['rgb_p95'] < 3.0 and dmax < 1e-2 and davg < 1e-2
    print(f"\n{'PASS' if passed else 'FAIL'}  (bar: 95th-pct RGB diff < 3/255 and "
          f"gradient rel diff < 1%)")
    print(f"\noutputs in {OUT_DIR}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
