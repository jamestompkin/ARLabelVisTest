"""End-to-end sanity check of the rendering pipeline.

Generates a CIELAB farthest-color LUT via the pure-Python Euclidean-over-hull
branch (no MATLAB, no neural bounding), writes it in the paired-text-file
format, loads it through `viz.load_lut`, and renders a 3D sRGB cube with
`viz.render_srgb_cube_isometric`. Intended to be runnable from a fresh clone
in under a minute.

Output PNG lands at `results/e2e_demo/lut_cube_cielab_euclidean.png`.
"""
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np

from arlabelvis.meshing import generate_LABs
from arlabelvis.distances import furthest_euclidean_lab_points
from arlabelvis.viz import load_lut, render_srgb_cube_isometric


OUT_DIR = Path("results/e2e_demo")

_timings: list[tuple[str, float]] = []


@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _timings.append((name, dt))
    print(f"    ({dt*1000:8.1f} ms)  {name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    interval = 8   # 33^3 ~= 35K input points; hull computation is fast
    print(f"[1/4] generating input grid at stepSize={interval}")
    with stage("generate_LABs"):
        allRGBs, allLABs = generate_LABs(stepSize=interval)

    print(f"      {len(allRGBs)} input points, CIELAB span "
          f"L in [{allLABs[:,0].min():.1f}, {allLABs[:,0].max():.1f}]  "
          f"a in [{allLABs[:,1].min():.1f}, {allLABs[:,1].max():.1f}]  "
          f"b in [{allLABs[:,2].min():.1f}, {allLABs[:,2].max():.1f}]")

    print("[2/4] computing farthest color (CIELAB, Euclidean over convex hull)")
    with stage("furthest_euclidean_lab_points"):
        furthest_rgb = furthest_euclidean_lab_points(allLABs)
    with stage("clip+cast to uint8"):
        furthest_rgb = np.clip(furthest_rgb, 0, 255).astype(np.uint8)

    lab_path = OUT_DIR / "AllCandidateLABvals_CIELAB_8_Euclidean.txt"
    rgb_path = OUT_DIR / "AllCorrespondingRGBVals_CIELAB_8_Euclidean.txt"
    print("[3/4] writing sparse LUT text files")
    with stage("np.savetxt (both files)"):
        np.savetxt(lab_path, furthest_rgb, delimiter=",", fmt="%d")
        np.savetxt(rgb_path, allRGBs, delimiter=",", fmt="%d")

    print("[4/4] loading LUT and rendering cube")
    with stage("load_lut (parse + 256^3 nearest-neighbor fill)"):
        lut = load_lut(lab_path, rgb_path, value="rgb")

    out_png = OUT_DIR / "lut_cube_cielab_euclidean.png"
    with stage("render_srgb_cube_isometric (PyVista, 3 cell-grid faces)"):
        render_srgb_cube_isometric(
            lut, save_path=out_png,
            title="CIELAB Euclidean farthest-color LUT (interval=8)",
        )

    # ---- Summary ----
    total = sum(dt for _, dt in _timings)
    print("\n=== timing summary (descending) ===")
    for name, dt in sorted(_timings, key=lambda x: -x[1]):
        print(f"  {dt*1000:8.1f} ms  {dt/total*100:5.1f} %   {name}")
    print(f"  {total*1000:8.1f} ms  100.0 %   TOTAL")


if __name__ == "__main__":
    main()
