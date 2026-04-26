"""Animate the convex-hull candidate set as the RGD regulariser alpha_hat varies.

alpha_hat is a property of the *metric* (RGD), not the candidate-set *geometry* —
RGD applies equally to the hull, sphere, or neural meshes. This script
isolates the metric axis: hull mesh fixed, sweep alpha_hat.

For each frame, ``compute_all_pairs_argmax`` runs RGD on the hull at the
current alpha_hat to find each hull vertex's farthest hull vertex. Inputs map
to their nearest hull vertex (in CIELAB) and inherit that vertex's
argmax output sRGB.

    alpha_hat --> per-vertex RGD argmax (all-pairs ADMM)
        --> per-input vertex (NN in CIELAB)
        --> per-input output sRGB

Three panels per frame:

1. 3D CIELAB scatter — gamut point cloud (faint) + hull wireframe + hull
   vertices coloured by their argmax output sRGB. The "what does each
   hull vertex map to" view.
2. Input partition (a*, b*) — input voxels coloured by output sRGB.
3. Pushforward (log-log) — K, K_eff, per-rank vlines.

Usage::

    uv run python -m scripts.paper.animate_alpha_hull --mp4
    uv run python -m scripts.paper.animate_alpha_hull --frames 60 --amin 0.01 --amax 2.0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

import multiprocessing as mp
import os

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import build_candidates, _nearest_rgb
from arlabelvis.meshing import generate_input_grid
from scripts.paper._paths import fig_path


def _subdivide_mesh_once(verts: np.ndarray, faces: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """One round of linear midpoint subdivision: each triangle -> 4 sub-triangles.

    Edge midpoints are deduplicated across shared edges, so the result stays
    watertight. New vertices are convex combinations of original vertices, so
    on a convex polytope mesh every midpoint lies on the original surface.
    """
    new_verts = [v.copy() for v in verts]
    edge_mid: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_mid:
            return edge_mid[key]
        idx = len(new_verts)
        new_verts.append(0.5 * (verts[a] + verts[b]))
        edge_mid[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        m01 = midpoint(v0, v1)
        m12 = midpoint(v1, v2)
        m20 = midpoint(v2, v0)
        new_faces.extend([
            (v0, m01, m20),
            (v1, m12, m01),
            (v2, m20, m12),
            (m01, m12, m20),
        ])
    return np.asarray(new_verts), np.asarray(new_faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Long-lived worker pool: build_mesh_ops once per worker, alpha_hat per task.
# Eliminates the 60x pool-spawn cost (~10s each on Windows) of repeatedly
# calling compute_all_pairs_argmax — the hull mesh is fixed for the whole
# sweep, so its mesh operators are too.
# ---------------------------------------------------------------------------
_OPS_GLOBAL = None


def _init_alpha_worker(V_bytes: bytes, V_shape: tuple,
                        F_bytes: bytes, F_shape: tuple) -> None:
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
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    from arlabelvis.rgd.mesh_ops import build_mesh_ops
    V = np.frombuffer(V_bytes, dtype=np.float64).reshape(V_shape)
    F = np.frombuffer(F_bytes, dtype=np.int64).reshape(F_shape)
    global _OPS_GLOBAL
    _OPS_GLOBAL = build_mesh_ops(V, F)


def _solve_alpha_task(args: tuple) -> tuple:
    """args = (frame_idx, source_idx, alpha_hat).
    Returns (frame_idx, source_idx, argmax_idx).
    """
    from arlabelvis.rgd.admm import rdg_admm
    fi, src, a = args
    u, _ = rdg_admm(_OPS_GLOBAL, x0=int(src), alpha_hat=float(a))
    return (int(fi), int(src), int(np.argmax(u)))

_log = logging.getLogger(__name__)


def _palette_from_per_seed_rgb(per_seed_rgb: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    flat = per_seed_rgb.astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    keys, counts = np.unique(packed, return_counts=True)
    palette = np.stack([(keys >> 16) & 0xFF,
                        (keys >> 8) & 0xFF,
                        keys & 0xFF], axis=1).astype(np.uint8)
    order = np.argsort(-counts)
    return palette[order], counts[order]


def _effective_palette_size(counts: np.ndarray) -> float:
    p = counts.astype(np.float64) / counts.sum()
    return float(1.0 / np.sum(p * p))


def _draw_frame(axes: tuple, *, alpha_hat: float,
                 hull_verts: np.ndarray, hull_edges: np.ndarray,
                 hull_vert_out_rgb: np.ndarray,
                 seed_cielab: np.ndarray, out_rgb: np.ndarray,
                 dense_cielab: np.ndarray, dense_rgb: np.ndarray,
                 hist_x_max: int) -> None:
    ax_3d, ax_2d, ax_hist = axes
    for ax in axes:
        ax.cla()

    palette, counts = _palette_from_per_seed_rgb(out_rgb)
    K = len(palette)
    K_eff = _effective_palette_size(counts)

    # --- panel 1: 3D gamut + hull wireframe + per-vertex argmax colour ---
    ax_3d.scatter(dense_cielab[:, 1], dense_cielab[:, 2], dense_cielab[:, 0],
                  c=dense_rgb, s=4, alpha=0.30, edgecolors="none")
    seg = hull_verts[hull_edges][:, :, [1, 2, 0]]
    lc = Line3DCollection(seg, colors="0.25", linewidths=0.7, alpha=0.45)
    ax_3d.add_collection3d(lc)
    ax_3d.scatter(hull_verts[:, 1], hull_verts[:, 2], hull_verts[:, 0],
                  c=hull_vert_out_rgb.astype(np.float32) / 255.0,
                  s=40, edgecolors="black", linewidths=0.4, alpha=0.95,
                  zorder=10)
    ax_3d.set_xlim(-100, 100); ax_3d.set_ylim(-100, 100); ax_3d.set_zlim(0, 100)
    ax_3d.set_xlabel("a*"); ax_3d.set_ylabel("b*"); ax_3d.set_zlabel("L*")
    ax_3d.view_init(elev=20, azim=-55)
    ax_3d.set_title(f"hull vertex partition\n"
                    f"$\\hat{{\\alpha}}$ = {alpha_hat:.3f}",
                    fontsize=10)

    # --- panel 2: input partition (a*, b*) coloured by output ---
    ax_2d.scatter(seed_cielab[:, 1], seed_cielab[:, 2],
                  c=out_rgb.astype(np.float32) / 255.0,
                  s=10, edgecolors="none", alpha=0.85)
    ax_2d.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax_2d.axvline(0, color="lightgray", lw=0.5, zorder=0)
    ax_2d.set_xlim(-100, 100); ax_2d.set_ylim(-100, 100)
    ax_2d.set_aspect("equal")
    ax_2d.set_xlabel("input a*"); ax_2d.set_ylabel("input b*")
    ax_2d.set_title(f"input partition  K = {K},  $K_{{\\mathrm{{eff}}}}$ = {K_eff:.0f}",
                    fontsize=10)

    # --- panel 3: pushforward ---
    ranks = np.arange(1, K + 1)
    frac = counts / counts.sum()
    marker_colors = palette.astype(np.float32) / 255.0
    ax_hist.vlines(ranks, ymin=1e-9, ymax=frac,
                   colors=marker_colors, linewidth=1.2)
    ax_hist.axvline(K_eff, color="k", lw=0.8, ls="--", alpha=0.7)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(0.7, max(hist_x_max, 10))
    ax_hist.set_ylim(1e-7, 1.0)
    ax_hist.set_xlabel("palette entry (rank, log)")
    ax_hist.set_ylabel("voxel fraction (log)")
    ax_hist.set_title(f"pushforward (K = {K},  $K_{{\\mathrm{{eff}}}}$ = {K_eff:.0f})",
                     fontsize=10)


def _hull_unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                       axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=60,
                   help="number of alpha_hat values to sweep")
    p.add_argument("--amin", type=float, default=0.01,
                   help="minimum alpha_hat (small = sharp, near-Euclidean)")
    p.add_argument("--amax", type=float, default=2.0,
                   help="maximum alpha_hat (large = strong RGD smoothing)")
    p.add_argument("--log-spaced", action="store_true",
                   help="sweep alpha_hat in log space (more resolution at low end)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None,
                   help="output path (defaults to figures/alpha_hull_animation/"
                        "alpha_hull.gif or .mp4 with --mp4)")
    p.add_argument("--mp4", action="store_true")
    p.add_argument("--n-workers", type=int, default=None,
                   help="RGD ADMM pool size (default os.cpu_count() - 1)")
    p.add_argument("--hull-subdivisions", type=int, default=2,
                   help="rounds of linear midpoint subdivision applied to "
                        "the bare 166-vert hull. 0 = bare hull (RGD ~ "
                        "Euclidean). 1 ~ 658 verts, 2 ~ 2624 verts, 3 ~ "
                        "10498 verts. Each round 4x's faces and ~4x's "
                        "ADMM cost.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out) if args.out else fig_path(
        "alpha_hull_animation",
        "alpha_hull.mp4" if args.mp4 else "alpha_hull.gif",
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- precompute static data: seed grid, hull, dense gamut backdrop ---
    print("[anim] sampling seed grid (interval=16)...", flush=True)
    seed_rgb_int, _ = generate_input_grid("sRGB", interval=16)
    seed_cielab = srgb_to_lab(seed_rgb_int.astype(np.float64))
    seed_rgb_u8 = seed_rgb_int.astype(np.uint8)

    print("[anim] sampling dense gamut for 3D backdrop (interval=8)...",
          flush=True)
    dense_rgb_int, _ = generate_input_grid("sRGB", interval=8)
    rng = np.random.default_rng(0)
    if len(dense_rgb_int) > 4000:
        idx = rng.choice(len(dense_rgb_int), size=4000, replace=False)
        dense_rgb_int = dense_rgb_int[idx]
    dense_cielab = srgb_to_lab(dense_rgb_int.astype(np.float64))
    dense_rgb_unit = dense_rgb_int.astype(np.float32) / 255.0

    print("[anim] building convex hull on CIELAB-mapped sRGB gamut...",
          flush=True)
    candidates = build_candidates(
        seed_cielab, seed_rgb_u8.astype(np.int64),
        shape="hull", smoothing="none",
        working_space="CIELAB", input_space="sRGB",
    )
    hull_verts = np.asarray(candidates.vertices)
    hull_faces = np.asarray(candidates.faces)
    print(f"[anim] bare hull: {len(hull_verts)} verts, {len(hull_faces)} faces",
          flush=True)
    for level in range(args.hull_subdivisions):
        hull_verts, hull_faces = _subdivide_mesh_once(hull_verts, hull_faces)
        print(f"[anim]   after subdiv {level + 1}: {len(hull_verts)} verts, "
              f"{len(hull_faces)} faces", flush=True)
    hull_rgbs = _nearest_rgb(hull_verts, seed_cielab,
                              seed_rgb_u8.astype(np.int64)).astype(np.uint8)
    hull_edges = _hull_unique_edges(hull_faces)
    print(f"[anim] final mesh: {len(hull_verts)} verts, {len(hull_faces)} faces, "
          f"{len(hull_edges)} unique edges", flush=True)

    input_to_vert = cKDTree(hull_verts).query(seed_cielab)[1]

    # --- alpha_hat schedule ---
    if args.log_spaced:
        alphas = np.geomspace(args.amin, args.amax, args.frames)
    else:
        alphas = np.linspace(args.amin, args.amax, args.frames)
    print(f"[anim] {args.frames} frames, alpha_hat {args.amin} -> {args.amax}",
          flush=True)

    # --- pre-pass: long-lived pool, build mesh ops ONCE per worker, then
    # submit (frame, source, alpha) for every (frame, source) pair. Workers
    # process tasks in any order; we route results back by frame_idx. ---
    nv = len(hull_verts)
    n_workers = args.n_workers or max(1, (os.cpu_count() or 1) - 1)
    V_arr = np.ascontiguousarray(hull_verts, dtype=np.float64)
    F_arr = np.ascontiguousarray(hull_faces, dtype=np.int64)
    tasks = [(fi, src, float(a))
             for fi, a in enumerate(alphas)
             for src in range(nv)]
    print(f"[anim] {len(tasks)} ADMM solves "
          f"({len(alphas)} alphas x {nv} sources), {n_workers} workers",
          flush=True)
    per_frame_vert_argmax = np.full((len(alphas), nv), -1, dtype=np.int64)

    import time
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_alpha_worker,
                  initargs=(V_arr.tobytes(), V_arr.shape,
                            F_arr.tobytes(), F_arr.shape)) as pool:
        chunksize = max(1, len(tasks) // (n_workers * 16))
        for n_done, (fi, src, am) in enumerate(
            pool.imap_unordered(_solve_alpha_task, tasks, chunksize=chunksize), 1
        ):
            per_frame_vert_argmax[fi, src] = am
            if n_done % 100 == 0 or n_done == len(tasks):
                dt = time.perf_counter() - t0
                rate = n_done / dt
                eta = (len(tasks) - n_done) / rate if rate > 0 else float("inf")
                print(f"  {n_done}/{len(tasks)}  ({rate:.1f} solves/s, "
                      f"ETA {eta / 60:.1f} min)", flush=True)
    print(f"[anim] all-pairs done in {(time.perf_counter() - t0) / 60:.1f} min",
          flush=True)

    per_frame_out_rgb = [hull_rgbs[per_frame_vert_argmax[fi][input_to_vert]]
                         for fi in range(len(alphas))]
    per_frame_hull_argmax = [hull_rgbs[per_frame_vert_argmax[fi]]
                             for fi in range(len(alphas))]
    ks = [len(_palette_from_per_seed_rgb(out_rgb)[0])
          for out_rgb in per_frame_out_rgb]
    hist_x_max = max(ks)
    print(f"[anim] K per frame = {ks}", flush=True)
    print(f"[anim] shared hist x-max = {hist_x_max}", flush=True)

    # --- render frames ---
    fig = plt.figure(figsize=(13.5, 4.6), dpi=96)
    ax_3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_2d = fig.add_subplot(1, 3, 2)
    ax_hist = fig.add_subplot(1, 3, 3)
    fig.tight_layout()

    def update(frame_idx):
        if (frame_idx % 10) == 0 or frame_idx == args.frames - 1:
            print(f"  render frame {frame_idx + 1}/{args.frames}  "
                  f"alpha_hat={alphas[frame_idx]:.4f}", flush=True)
        _draw_frame(
            (ax_3d, ax_2d, ax_hist),
            alpha_hat=float(alphas[frame_idx]),
            hull_verts=hull_verts, hull_edges=hull_edges,
            hull_vert_out_rgb=per_frame_hull_argmax[frame_idx],
            seed_cielab=seed_cielab,
            out_rgb=per_frame_out_rgb[frame_idx],
            dense_cielab=dense_cielab, dense_rgb=dense_rgb_unit,
            hist_x_max=hist_x_max,
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=args.frames,
        interval=1000.0 / args.fps, blit=False,
    )
    if args.mp4:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
