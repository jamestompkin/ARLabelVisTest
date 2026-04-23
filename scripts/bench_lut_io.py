"""Benchmark text vs binary I/O for a production-sized (256^3 = 16.7M entries) LUT.

Measures:
- Save: current interpolate_interval per-line Python loop (what the pipeline does today)
        vs np.savetxt in one call  vs np.save (.npy).
- Load: np.loadtxt (current)       vs np.load (.npy).

Only writes/reads dense 256^3 LAB values (one row per voxel). No semantic interpretation.
"""
from pathlib import Path
import time
import numpy as np

OUT = Path("results/bench_lut_io")
OUT.mkdir(parents=True, exist_ok=True)

N = 256
# Fake LAB-style values: L in [0,100], a,b in [-128, 127]. Float32 to keep memory light.
rng = np.random.default_rng(0)
lut_lab = np.empty((N, N, N, 3), dtype=np.float32)
lut_lab[..., 0] = rng.uniform(0, 100, (N, N, N))
lut_lab[..., 1] = rng.uniform(-128, 127, (N, N, N))
lut_lab[..., 2] = rng.uniform(-128, 127, (N, N, N))
flat = lut_lab.reshape(-1, 3)
print(f"Array shape {lut_lab.shape}  dtype {lut_lab.dtype}  size {lut_lab.nbytes / 1e6:.1f} MB")


def time_it(label, fn):
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    print(f"  {dt*1000:10.0f} ms   {label}")
    return dt, res


print("\n--- SAVE ---")
txt_line_path = OUT / "dense_lut_per_line.txt"
def save_per_line():
    with txt_line_path.open("w") as f:
        for row in flat:
            f.write(f"{float(row[0])},{float(row[1])},{float(row[2])}\n")

txt_savetxt_path = OUT / "dense_lut_savetxt.txt"
npy_path = OUT / "dense_lut.npy"

dt_per_line, _ = time_it("current: Python-loop per-line write", save_per_line)
dt_savetxt, _ = time_it("np.savetxt (text, one call)",
                        lambda: np.savetxt(txt_savetxt_path, flat, delimiter=",", fmt="%.6f"))
dt_npy, _ = time_it("np.save (.npy binary, float32)",
                    lambda: np.save(npy_path, lut_lab))

print(f"\nFile sizes:")
for p in (txt_line_path, txt_savetxt_path, npy_path):
    print(f"  {p.stat().st_size / 1e6:8.1f} MB   {p.name}")

print("\n--- LOAD ---")
dt_loadtxt, _ = time_it("np.loadtxt from Python-loop file",
                        lambda: np.loadtxt(txt_line_path, delimiter=",", dtype=np.float32))
dt_loadtxt2, _ = time_it("np.loadtxt from np.savetxt file",
                         lambda: np.loadtxt(txt_savetxt_path, delimiter=",", dtype=np.float32))
dt_load_npy, _ = time_it("np.load (.npy binary)",
                         lambda: np.load(npy_path))

print("\n--- SUMMARY ---")
print(f"  save:  per-line  {dt_per_line*1000:.0f} ms   "
      f"savetxt  {dt_savetxt*1000:.0f} ms   "
      f"npy  {dt_npy*1000:.0f} ms   "
      f"(speedup per-line -> npy:  {dt_per_line/dt_npy:.0f}x)")
print(f"  load:  loadtxt  {dt_loadtxt2*1000:.0f} ms    "
      f"np.load  {dt_load_npy*1000:.0f} ms  "
      f"(speedup: {dt_loadtxt2/dt_load_npy:.0f}x)")
