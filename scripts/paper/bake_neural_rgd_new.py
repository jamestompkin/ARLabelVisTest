"""Bake LUT_CIELAB_NEURAL_RGD_125 under the corrected (NEW) algorithm.

Standalone — does NOT touch arlabelvis/luts.py. Implements the orthogonal-
factoring algorithm end-to-end:

    h(c) = NN_to_admitted_sample( argmax_{v ∈ mesh ∩ admitted_mapped} u_RGD(v ; nearest_vert(c)) )

For each input gamut voxel c:
  1. Find nearest mesh vertex v_c
  2. Solve RGD from v_c → distance vector u over all mesh verts
  3. argmax of u restricted to mesh verts that are NN-targets of admitted gamut samples
  4. Output = the admitted gamut sample whose closest mesh vert was the argmax

Produces:
  - results/shortpaper/validation/neural_rgd_125_new_lut.npy  (256³×3 sRGB LUT)
  - results/shortpaper/validation/neural_rgd_125_old_vs_new.md  (report)

Compute: ~80 min for the all-pairs RGD step on a 14k-vert mesh.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import (
    LutConfig, get_lut, lut_to_srgb_u8, _nearest_rgb,
)
from arlabelvis.meshing import generate_input_grid
from arlabelvis.neural_bounding import (
    NeuralBoundingParams, train_gamut_mlp, mesh_from_mlp, mesh_from_mlp_mc,
)
from scripts.paper._configs import LUT_CIELAB_NEURAL_RGD_125
from scripts.paper._paths import SHORTPAPER_DIR

_log = logging.getLogger(__name__)


# Worker-pool globals for restricted-argmax ADMM solves.
_OPS_GLOBAL = None
_ALPHA_GLOBAL = None
_ADMITTED_IDX_GLOBAL = None  # mesh vert indices that are admitted-mapped


def _init_worker(V_bytes, V_shape, F_bytes, F_shape,
                  alpha_hat, admitted_idx_bytes, admitted_idx_shape):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    try:
        import torch as _t
        _t.set_num_threads(1)
        _t.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    V = np.frombuffer(V_bytes, dtype=np.float64).reshape(V_shape)
    F = np.frombuffer(F_bytes, dtype=np.int64).reshape(F_shape)
    admitted = np.frombuffer(admitted_idx_bytes, dtype=np.int64).reshape(
        admitted_idx_shape)
    global _OPS_GLOBAL, _ALPHA_GLOBAL, _ADMITTED_IDX_GLOBAL
    _OPS_GLOBAL = build_mesh_ops(V, F)
    _ALPHA_GLOBAL = alpha_hat
    _ADMITTED_IDX_GLOBAL = admitted


def _solve_restricted_argmax(src):
    from arlabelvis.rgd.admm import rdg_admm
    u, _ = rdg_admm(_OPS_GLOBAL, x0=int(src), alpha_hat=_ALPHA_GLOBAL)
    u_admitted = u[_ADMITTED_IDX_GLOBAL]
    return int(src), int(_ADMITTED_IDX_GLOBAL[int(np.argmax(u_admitted))])


def _max_gradient_and_std(lut_u8: np.ndarray) -> tuple[float, float]:
    f = lut_u8.astype(np.float32)
    gx, gy, gz = np.gradient(f, axis=(0, 1, 2))
    mag = np.sqrt(gx**2 + gy**2 + gz**2).sum(axis=-1)
    return float(mag.max()), float(mag.std())


def _mlp_eval_batched(mlp, affine, points_lab: np.ndarray,
                       batch: int = 1_000_000) -> np.ndarray:
    """Evaluate the MLP on ``points_lab`` in batches; return (N,) float scores."""
    out = np.empty(len(points_lab), dtype=np.float32)
    for lo in range(0, len(points_lab), batch):
        hi = min(len(points_lab), lo + batch)
        block = affine.to_unit(points_lab[lo:hi].astype(np.float32))
        with torch.no_grad():
            out[lo:hi] = mlp(torch.from_numpy(block)).cpu().numpy()
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = LutConfig(
        working_space=LUT_CIELAB_NEURAL_RGD_125.working_space,
        shape=LUT_CIELAB_NEURAL_RGD_125.shape,
        metric=LUT_CIELAB_NEURAL_RGD_125.metric,
        metric_rgd_alpha_hat=LUT_CIELAB_NEURAL_RGD_125.metric_rgd_alpha_hat,
        interval=1,  # full 256^3 dense
    )
    print(f"[bake-new] config: {cfg}", flush=True)
    out_dir = SHORTPAPER_DIR / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: dense input grid (16.7M sRGB voxels) → CIELAB ---
    print("[bake-new] stage 1: dense input grid (interval=1)...", flush=True)
    t0 = time.perf_counter()
    rgb_int, _ = generate_input_grid("sRGB", interval=1)
    print(f"  {len(rgb_int)} voxels  ({time.perf_counter()-t0:.1f}s)",
          flush=True)
    print("  converting to CIELAB...", flush=True)
    t0 = time.perf_counter()
    lab = srgb_to_lab(rgb_int.astype(np.float64))
    print(f"  CIELAB done  ({time.perf_counter()-t0:.1f}s)", flush=True)
    rgb_u8 = rgb_int.astype(np.uint8)

    # --- Stage 2: train MLP + extract mesh (same as production) ---
    print("[bake-new] stage 2: train MLP...", flush=True)
    nb_params = NeuralBoundingParams(
        bound_bias=cfg.shape_neural_bound_bias,
        iterations=cfg.shape_neural_iterations,
    )
    # Train on a sparser sample to avoid 16.7M-point training (matches what
    # production does — Liu's MLP is trained on the gamut sample, doesn't
    # need every voxel).
    t0 = time.perf_counter()
    train_pts, _ = generate_input_grid("sRGB", interval=16)
    train_lab = srgb_to_lab(train_pts.astype(np.float64))
    mlp, affine = train_gamut_mlp(train_lab.astype(np.float32), nb_params,
                                    device="cpu")
    print(f"  trained ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("[bake-new] stage 2b: extract mesh...", flush=True)
    t0 = time.perf_counter()
    mesh = mesh_from_mlp(mlp, affine, nb_params, device="cpu")
    if (len(mesh.faces) == 0 or not mesh.is_watertight
            or not mesh.is_winding_consistent):
        print("  polytope mesh non-manifold; falling back to MC", flush=True)
        mesh = mesh_from_mlp_mc(mlp, affine, nb_params, device="cpu")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"  mesh: {len(verts)} verts, {len(faces)} faces "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # --- Stage 3: admission via MLP threshold over all 16.7M voxels ---
    print("[bake-new] stage 3: admission predicate (MLP eval over 16.7M)...",
          flush=True)
    t0 = time.perf_counter()
    mlp_vals = _mlp_eval_batched(mlp, affine, lab)
    admit = mlp_vals >= 0.5
    n_admit = int(admit.sum())
    print(f"  admitted: {n_admit:,} / {len(lab):,} "
          f"({100*n_admit/len(lab):.1f}%)  ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # --- Stage 4: map admitted voxels → mesh verts; build admitted_mesh_verts ---
    print("[bake-new] stage 4: map admitted voxels to mesh verts...",
          flush=True)
    t0 = time.perf_counter()
    tree_verts = cKDTree(verts)
    # Map every voxel (admitted or not) to its nearest mesh vert — used
    # later for both the admitted set AND for routing inputs to sources.
    voxel_to_vert = tree_verts.query(lab)[1].astype(np.int64)
    print(f"  voxel-to-vert mapping done ({time.perf_counter()-t0:.1f}s)",
          flush=True)
    admitted_mesh_verts = np.unique(voxel_to_vert[admit])
    print(f"  {len(admitted_mesh_verts):,} unique admitted-mapped mesh verts",
          flush=True)

    # For each mesh vert, the "closest admitted voxel that maps to it" — used
    # to produce the output sRGB once we choose an argmax mesh vert. We'll
    # populate this lazily from voxel_to_vert + admit + lab.
    print("  building per-mesh-vert closest-admitted-voxel index...",
          flush=True)
    t0 = time.perf_counter()
    # For each mesh vert v, find the admitted voxel mapping to v with smallest
    # ||lab - verts[v]|| in CIELAB (== smallest distance, since voxel_to_vert
    # already picked the nearest, but multiple voxels can map to same v).
    # Easiest: for each admitted voxel, store (vert, voxel_idx, dist²); then
    # take argmin per vert via np.minimum.at-equivalent.
    admitted_voxel_idx = np.flatnonzero(admit)
    admitted_voxel_vert = voxel_to_vert[admitted_voxel_idx]
    diff = lab[admitted_voxel_idx] - verts[admitted_voxel_vert]
    d2 = np.sum(diff * diff, axis=1)
    # For each mesh vert that has at least one admitted voxel, the
    # closest-admitted-voxel index. Initialize to -1 (none).
    closest_admitted_voxel_per_vert = np.full(len(verts), -1, dtype=np.int64)
    best_d2_per_vert = np.full(len(verts), np.inf, dtype=np.float64)
    # Vectorized "min by group" via sorting then taking first per group.
    order = np.argsort(d2, kind="stable")
    sorted_vert = admitted_voxel_vert[order]
    sorted_voxel = admitted_voxel_idx[order]
    sorted_d2 = d2[order]
    seen = np.zeros(len(verts), dtype=bool)
    for vert, voxel, d in zip(sorted_vert, sorted_voxel, sorted_d2):
        if not seen[vert]:
            seen[vert] = True
            closest_admitted_voxel_per_vert[vert] = voxel
            best_d2_per_vert[vert] = d
    n_with_admitted = int(seen.sum())
    print(f"  {n_with_admitted:,} mesh verts have ≥1 admitted voxel mapped "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # --- Stage 5: parallel restricted-argmax RGD over all mesh verts ---
    print(f"[bake-new] stage 5: RGD all-pairs with restricted argmax "
          f"({len(verts)} sources)...", flush=True)
    n_workers = max(1, (os.cpu_count() or 1) - 1)
    V_arr = np.ascontiguousarray(verts, dtype=np.float64)
    F_arr = np.ascontiguousarray(faces, dtype=np.int64)
    admitted_idx_arr = np.ascontiguousarray(admitted_mesh_verts, dtype=np.int64)
    new_argmax_per_vert = np.full(len(verts), -1, dtype=np.int64)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    sources = list(range(len(verts)))
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(V_arr.tobytes(), V_arr.shape,
                            F_arr.tobytes(), F_arr.shape,
                            float(cfg.metric_rgd_alpha_hat),
                            admitted_idx_arr.tobytes(),
                            admitted_idx_arr.shape)) as pool:
        chunksize = max(1, len(sources) // (n_workers * 8))
        for n_done, (src, am) in enumerate(
            pool.imap_unordered(_solve_restricted_argmax, sources,
                                 chunksize=chunksize), 1
        ):
            new_argmax_per_vert[src] = am
            if n_done % 200 == 0 or n_done == len(sources):
                dt = time.perf_counter() - t0
                rate = n_done / dt
                eta = (len(sources) - n_done) / rate if rate > 0 else 0
                print(f"    [{n_done:>5}/{len(sources):>5}]  {rate:.1f} src/s, "
                      f"ETA {eta/60:.1f} min", flush=True)
    print(f"  RGD done ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    # --- Stage 6: route every input voxel → output sRGB ---
    print("[bake-new] stage 6: building dense LUT...", flush=True)
    t0 = time.perf_counter()
    chosen_vert_per_voxel = new_argmax_per_vert[voxel_to_vert]
    chosen_admitted_voxel = closest_admitted_voxel_per_vert[chosen_vert_per_voxel]
    # If a chosen mesh vert had no admitted voxel mapped (shouldn't happen,
    # but guard anyway), fall back to the chosen mesh vert's NN-snap to seeds.
    fallback_mask = chosen_admitted_voxel < 0
    out_rgb = np.empty_like(rgb_u8)
    out_rgb[~fallback_mask] = rgb_u8[chosen_admitted_voxel[~fallback_mask]]
    if fallback_mask.any():
        # Should not happen because every admitted_mesh_vert by construction
        # has ≥1 admitted voxel mapped, and chosen_vert_per_voxel comes from
        # new_argmax_per_vert which IS admitted by construction.
        n_fb = int(fallback_mask.sum())
        print(f"  WARN: {n_fb:,} voxels hit fallback path; using mesh-vert "
              f"NN-snap to seeds for those.", flush=True)
        train_rgb = train_pts.astype(np.uint8)
        vert_seed_rgb = _nearest_rgb(verts, train_lab,
                                       train_pts.astype(np.int64)).astype(np.uint8)
        out_rgb[fallback_mask] = vert_seed_rgb[
            chosen_vert_per_voxel[fallback_mask]
        ]
    new_lut = out_rgb.reshape(256, 256, 256, 3).astype(np.uint8)
    print(f"  dense LUT built ({time.perf_counter()-t0:.1f}s)", flush=True)

    new_lut_path = out_dir / "neural_rgd_125_new_lut.npy"
    np.save(new_lut_path, new_lut)
    print(f"  saved {new_lut_path}", flush=True)

    # --- Stage 7: load OLD LUT (cached or rebuild), compute metrics + diff ---
    print("[bake-new] stage 7: load OLD LUT for comparison...", flush=True)
    t0 = time.perf_counter()
    cfg_old = LutConfig(
        working_space=LUT_CIELAB_NEURAL_RGD_125.working_space,
        shape=LUT_CIELAB_NEURAL_RGD_125.shape,
        metric=LUT_CIELAB_NEURAL_RGD_125.metric,
        metric_rgd_alpha_hat=LUT_CIELAB_NEURAL_RGD_125.metric_rgd_alpha_hat,
    )
    old_raw = get_lut(cfg_old)
    old_lut = lut_to_srgb_u8(old_raw, cfg_old.output_space)
    print(f"  OLD LUT loaded ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("[bake-new] computing metrics...", flush=True)
    new_max, new_std = _max_gradient_and_std(new_lut)
    old_max, old_std = _max_gradient_and_std(old_lut)
    n_voxels = new_lut.size // 3
    diff_voxels = (new_lut.reshape(-1, 3) != old_lut.reshape(-1, 3)).any(axis=1).sum()
    rgb_diff = np.abs(new_lut.astype(np.int32) - old_lut.astype(np.int32))
    mean_rgb_diff = float(rgb_diff.sum(axis=-1).mean())
    max_rgb_diff = int(rgb_diff.sum(axis=-1).max())

    report = (out_dir / "neural_rgd_125_old_vs_new.md")
    report.write_text(f"""# `LUT_CIELAB_NEURAL_RGD_125` — OLD vs NEW algorithm

Baked under both algorithms at the full 256³ dense LUT resolution. NEW
implements the orthogonal-axis framing: admit MLP-inside gamut samples,
restrict argmax to admitted-mapped mesh verts, output is the closest
admitted gamut sample whose mesh vert was the argmax.

## Mesh

- Vertices: {len(verts):,}
- Faces: {len(faces):,}
- Admitted gamut voxels: {n_admit:,} / {len(lab):,} ({100*n_admit/len(lab):.1f}%)
- Mesh verts that received ≥1 admitted voxel: {n_with_admitted:,} / {len(verts):,}

## Per-voxel comparison

- Voxels where output differs: **{diff_voxels:,} / {n_voxels:,}** ({100*diff_voxels/n_voxels:.2f}%)
- Mean per-voxel ‖Δsrgb‖₁: {mean_rgb_diff:.2f}
- Max per-voxel ‖Δsrgb‖₁: {max_rgb_diff}

## Gradient metrics on the dense LUT

| Algorithm | max-gradient | std |
|---|---|---|
| OLD | {old_max:.2f} | {old_std:.2f} |
| NEW | {new_max:.2f} | {new_std:.2f} |
| Δ (NEW − OLD) | {new_max - old_max:+.2f} | {new_std - old_std:+.2f} |

## Paper headline (for reference)

The thesis abstract reports max-gradient 217.95 → 41.60 and std 23.40 →
6.79 for the same recipe (Kwon's ΔE₀₀ baseline → ours: neural+RGD α̂=1.25).
Those numbers came from OLD's bake-out. Under NEW, the "ours" entry of
that comparison would be ({new_max:.2f}, {new_std:.2f}). If publishing
NEW, the abstract should be updated to "{old_max:.2f} (OLD: {old_max:.2f}) → {new_max:.2f}".

## Files

- `neural_rgd_125_new_lut.npy` — NEW dense LUT (256³×3 uint8)
- OLD dense LUT lives in the LutCache under key `{cfg_old.key()}`
""")
    print(f"\n=== REPORT ===\n", flush=True)
    print(report.read_text(), flush=True)
    print(f"\nWrote {report}", flush=True)


if __name__ == "__main__":
    main()
