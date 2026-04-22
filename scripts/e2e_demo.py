"""End-to-end sanity check of the rendering pipeline.

Generates a CIELAB farthest-color LUT via the pure-Python Euclidean-over-hull
branch (no MATLAB, no neural bounding), writes it in the paired-text-file
format, loads it through `viz.load_lut`, and renders a 3D RGB cube with
`viz.render_rgb_cube_isometric`. Intended to be runnable from a fresh clone
in under a minute.

Output PNG lands at `results/e2e_demo/lut_cube_cielab_euclidean.png`.
"""
from pathlib import Path
import numpy as np

from arlabelvis.meshing import generate_LABs
from arlabelvis.distances import furthest_euclidean_lab_points
from arlabelvis.viz import load_lut, render_rgb_cube_isometric


OUT_DIR = Path("results/e2e_demo")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    interval = 8   # 33^3 ~= 35K input points; hull computation is fast
    print(f"[1/4] generating input grid at stepSize={interval}...")
    allRGBs, allLABs = generate_LABs(stepSize=interval)
    print(f"      {len(allRGBs)} input points, CIELAB span "
          f"L∈[{allLABs[:,0].min():.1f}, {allLABs[:,0].max():.1f}]  "
          f"a∈[{allLABs[:,1].min():.1f}, {allLABs[:,1].max():.1f}]  "
          f"b∈[{allLABs[:,2].min():.1f}, {allLABs[:,2].max():.1f}]")

    print("[2/4] computing farthest color (CIELAB, Euclidean over convex hull)...")
    furthest_rgb = furthest_euclidean_lab_points(allLABs)  # (N,3) float in [0,255]
    furthest_rgb = np.clip(furthest_rgb, 0, 255).astype(np.uint8)

    # Paired sparse-LUT text files (same format the rest of the pipeline uses).
    # Note: value="rgb" later, since furthest_euclidean_lab_points returns RGB,
    # not LAB. The filename still carries the "LABvals" prefix for consistency
    # with the rest of the codebase's naming convention.
    lab_path = OUT_DIR / "AllCandidateLABvals_CIELAB_8_Euclidean.txt"
    rgb_path = OUT_DIR / "AllCorrespondingRGBVals_CIELAB_8_Euclidean.txt"
    np.savetxt(lab_path, furthest_rgb, delimiter=",", fmt="%d")
    np.savetxt(rgb_path, allRGBs, delimiter=",", fmt="%d")
    print(f"[3/4] wrote sparse LUT files: {lab_path.name}, {rgb_path.name}")

    # Load back as dense 256^3 LUT (nearest-neighbor fill of the sparse grid)
    lut = load_lut(lab_path, rgb_path, value="rgb")

    out_png = OUT_DIR / "lut_cube_cielab_euclidean.png"
    print(f"[4/4] rendering LUT cube -> {out_png}")
    render_rgb_cube_isometric(
        lut, save_path=out_png,
        title="CIELAB Euclidean farthest-color LUT (interval=8)",
        stride=2,
    )
    print("done.")


if __name__ == "__main__":
    main()
