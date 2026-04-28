"""Compare two algorithm formulations on the paper's final recipe.

Final recipe: ``LUT_CIELAB_NEURAL_RGD_125`` (neural-bounded mesh, RGD α̂=1.25).

**OLD** (production today): argmax-RGD against all mesh verts → NN-snap of
the chosen mesh vertex to nearest seed sRGB.

**NEW** (orthogonal-axis framing): admit gamut samples where MLP(x) ≥ thresh.
For each input voxel, solve RGD from its nearest mesh vert; take argmax of u
restricted to the *admitted-gamut-mapped* mesh verts. Output = the admitted
gamut sample whose mesh vert won.

Reports K, K_eff, per-input diffs.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import LutConfig, _nearest_rgb
from arlabelvis.meshing import generate_input_grid
from arlabelvis.neural_bounding import (
    NeuralBoundingParams, train_gamut_mlp, mesh_from_mlp, mesh_from_mlp_mc,
)
from arlabelvis.rgd.allpairs import compute_all_pairs_argmax
from scripts.paper._configs import LUT_CIELAB_NEURAL_RGD_125

_log = logging.getLogger(__name__)


# --- Long-lived pool: build MeshOps once per worker, solve rdg_admm per task,
#     return the FULL u vector so callers can post-process restricted argmax. ---
_OPS_GLOBAL = None
_ALPHA_GLOBAL = None


def _init_worker(V_bytes: bytes, V_shape: tuple,
                  F_bytes: bytes, F_shape: tuple, alpha_hat: float) -> None:
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
    global _OPS_GLOBAL, _ALPHA_GLOBAL
    _OPS_GLOBAL = build_mesh_ops(V, F)
    _ALPHA_GLOBAL = alpha_hat


def _solve_one_full_u(src: int) -> tuple:
    """Return (src, u) — full RGD distance vector from source vertex src."""
    from arlabelvis.rgd.admm import rdg_admm
    u, _ = rdg_admm(_OPS_GLOBAL, x0=int(src), alpha_hat=_ALPHA_GLOBAL)
    return int(src), u


def _palette_keys(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.astype(np.uint32)
    return (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]


def _summary(label: str, out_rgb_per_input: np.ndarray) -> dict:
    keys, counts = np.unique(_palette_keys(out_rgb_per_input), return_counts=True)
    K = len(keys)
    p = counts.astype(np.float64) / counts.sum()
    K_eff = float(1.0 / np.sum(p * p))
    return {"label": label, "K": K, "K_eff": K_eff}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n-workers", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = LutConfig(
        working_space=LUT_CIELAB_NEURAL_RGD_125.working_space,
        shape=LUT_CIELAB_NEURAL_RGD_125.shape,
        metric=LUT_CIELAB_NEURAL_RGD_125.metric,
        metric_rgd_alpha_hat=LUT_CIELAB_NEURAL_RGD_125.metric_rgd_alpha_hat,
        interval=args.interval,
    )
    print(f"[validate] cfg: shape={cfg.shape}, metric={cfg.metric}, "
          f"alpha_hat={cfg.metric_rgd_alpha_hat}, interval={args.interval}",
          flush=True)

    print("[validate] building seed grid + working-space points...", flush=True)
    input_pts, srgb_pts = generate_input_grid(cfg.input_space, interval=args.interval)
    all_rgbs = np.clip(np.round(srgb_pts), 0, 255).astype(np.int64)
    seed_lab = srgb_to_lab(input_pts.astype(np.float64))
    print(f"  {len(seed_lab)} seeds", flush=True)

    print("[validate] training MLP...", flush=True)
    nb_params = NeuralBoundingParams(
        bound_bias=cfg.shape_neural_bound_bias,
        iterations=cfg.shape_neural_iterations,
    )
    t0 = time.perf_counter()
    mlp, affine = train_gamut_mlp(seed_lab.astype(np.float32), nb_params,
                                    device="cpu")
    print(f"  training: {time.perf_counter() - t0:.1f}s", flush=True)

    print("[validate] extracting mesh...", flush=True)
    t0 = time.perf_counter()
    mesh = mesh_from_mlp(mlp, affine, nb_params, device="cpu")
    if (len(mesh.faces) == 0 or not mesh.is_watertight
            or not mesh.is_winding_consistent):
        print("  polytope mesh non-manifold; falling back to MC", flush=True)
        mesh = mesh_from_mlp_mc(mlp, affine, nb_params, device="cpu")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    print(f"  mesh: {len(verts)} verts, {len(faces)} faces "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    vert_rgb = _nearest_rgb(verts, seed_lab, all_rgbs).astype(np.uint8)

    # --- OLD: argmax-RGD against full mesh, NN-snap output ---
    print("[validate] OLD: argmax-RGD against full mesh + NN-snap...", flush=True)
    t0 = time.perf_counter()
    per_vert_argmax = compute_all_pairs_argmax(
        verts, faces, cfg.metric_rgd_alpha_hat,
        n_workers=args.n_workers,
        one_indexed=False, progress_every=200,
    ).astype(np.int64)
    input_to_vert = cKDTree(verts).query(seed_lab)[1]
    old_out = vert_rgb[per_vert_argmax[input_to_vert]]
    print(f"  OLD: {time.perf_counter() - t0:.1f}s", flush=True)

    # --- NEW: admit MLP-inside gamut, restrict argmax ---
    print("[validate] NEW: building admission predicate + restricted argmax...",
          flush=True)
    pts_unit = affine.to_unit(seed_lab.astype(np.float32))
    with torch.no_grad():
        mlp_vals = mlp(torch.from_numpy(pts_unit)).cpu().numpy()
    admit = mlp_vals >= args.threshold
    n_admit = int(admit.sum())
    print(f"  admitted: {n_admit} / {len(seed_lab)} "
          f"({100*n_admit/len(seed_lab):.1f}%)", flush=True)
    if n_admit == 0:
        print("  empty admission set; aborting NEW", flush=True)
        return

    admit_lab = seed_lab[admit]
    admit_to_vert = cKDTree(verts).query(admit_lab)[1]
    admitted_mask = np.zeros(len(verts), dtype=bool)
    admitted_mask[np.unique(admit_to_vert)] = True
    admitted_idx = np.flatnonzero(admitted_mask)
    print(f"  admitted-mapped mesh verts: {len(admitted_idx)}", flush=True)

    # Sources: each unique input-mesh-vert (one ADMM per source).
    unique_input_verts, inverse = np.unique(input_to_vert, return_inverse=True)
    print(f"  unique source verts to solve: {len(unique_input_verts)}",
          flush=True)

    # Parallel: long-lived pool, full-u vector per source, restrict argmax client-side.
    n_workers = args.n_workers or max(1, (os.cpu_count() or 1) - 1)
    V_arr = np.ascontiguousarray(verts, dtype=np.float64)
    F_arr = np.ascontiguousarray(faces, dtype=np.int64)
    print(f"  spawning {n_workers} workers...", flush=True)
    new_argmax_per_source = np.empty(len(unique_input_verts), dtype=np.int64)
    src_to_position = {int(s): i for i, s in enumerate(unique_input_verts)}

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(V_arr.tobytes(), V_arr.shape,
                            F_arr.tobytes(), F_arr.shape,
                            float(cfg.metric_rgd_alpha_hat))) as pool:
        chunksize = max(1, len(unique_input_verts) // (n_workers * 8))
        for n_done, (src, u) in enumerate(
            pool.imap_unordered(_solve_one_full_u,
                                 unique_input_verts.tolist(),
                                 chunksize=chunksize), 1
        ):
            u_admitted = u[admitted_idx]
            new_argmax_per_source[src_to_position[src]] = (
                admitted_idx[int(np.argmax(u_admitted))]
            )
            if n_done % 50 == 0 or n_done == len(unique_input_verts):
                dt = time.perf_counter() - t0
                rate = n_done / dt
                eta = (len(unique_input_verts) - n_done) / rate
                print(f"    [{n_done}/{len(unique_input_verts)}] "
                      f"{rate:.1f} src/s, ETA {eta/60:.1f} min", flush=True)
    print(f"  NEW: {(time.perf_counter() - t0)/60:.1f} min", flush=True)

    new_argmax_per_input = new_argmax_per_source[inverse]
    new_out = vert_rgb[new_argmax_per_input]

    # --- compare ---
    print("\n=== COMPARISON ===", flush=True)
    old = _summary("OLD", old_out)
    new = _summary("NEW", new_out)
    diffs = (old_out != new_out).any(axis=1).sum()
    print(f"OLD: K={old['K']:>3}, K_eff={old['K_eff']:.1f}", flush=True)
    print(f"NEW: K={new['K']:>3}, K_eff={new['K_eff']:.1f}", flush=True)
    print(f"per-input diffs: {diffs} / {len(seed_lab)} "
          f"({100*diffs/len(seed_lab):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
