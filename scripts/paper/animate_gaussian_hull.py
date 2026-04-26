"""Animate the hull-Gaussian-smoothed candidate mesh as σ varies.

σ is the *geometric* knob on the hull-Gaussian substrate: voxelise the dense
sRGB gamut in CIELAB, Gaussian-blur the indicator field, marching-cubes the
isosurface. Sweeping σ continuously deforms the candidate mesh from a
voxelised approximation of the convex hull (σ→0, staircased) to a softly
rounded blob (σ large).

Metric is fixed at Euclidean (L2 in CIELAB) so the only thing changing is
the candidate-mesh geometry. The chain is::

    σ --> blurred indicator field
       --> isosurface mesh (verts on the σ-smoothed boundary)
       --> per-input farthest mesh vertex (Euclidean L2)
       --> per-vertex nearest displayable sRGB
       --> per-input output sRGB

Three panels per frame, same layout as ``animate_alpha_hull.py``:

1. 3D CIELAB scatter — gamut backdrop + isosurface mesh wireframe + mesh
   vertices coloured by their nearest displayable sRGB. Geometric context.
2. Input partition (a*, b*) — input voxels coloured by output sRGB.
3. Pushforward (log-log) — K, K_eff, per-rank vlines.

Usage::

    uv run python -m scripts.paper.animate_gaussian_hull --mp4
    uv run python -m scripts.paper.animate_gaussian_hull --frames 60 --smin 0 --smax 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import _decimate_mesh, _nearest_rgb
from arlabelvis.meshing import generate_input_grid
from scripts.paper._paths import fig_path

_log = logging.getLogger(__name__)


def _voxelise_dense_gamut(vox: int = 256, padding: int = 2
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Voxelise the full sRGB gamut in CIELAB working space exactly once.

    Returns (grid, mins, voxel_size, padding) so per-frame blur+MC can reuse
    them. Mirrors the upfront stages of ``meshing.points_to_mesh`` but stops
    before the blur (so each σ frame supplies its own).
    """
    print("[anim] generating dense interval=1 sRGB grid...", flush=True)
    dense_rgb, _ = generate_input_grid("sRGB", interval=1)
    print(f"[anim] converting {len(dense_rgb)} points to CIELAB...", flush=True)
    dense_cielab = srgb_to_lab(dense_rgb.astype(np.float64))

    mins = dense_cielab.min(axis=0)
    maxs = dense_cielab.max(axis=0)
    ranges = maxs - mins + 1e-12
    longest = ranges.max()
    per_axis_res = np.round((ranges / longest) * vox).astype(int)
    per_axis_res = np.clip(per_axis_res, 16, vox)

    scale = (per_axis_res - 1 - 2 * padding) / ranges
    idx = np.floor((dense_cielab - mins) * scale).astype(np.int32) + padding
    idx = np.clip(idx, 0, per_axis_res - 1)

    grid = np.zeros(tuple(per_axis_res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    voxel_size = ranges / (per_axis_res - 1)
    print(f"[anim] voxel grid: {tuple(per_axis_res)}, "
          f"voxel_size={voxel_size.round(3)}", flush=True)
    return grid, mins, voxel_size, padding


def _mesh_at_sigma(grid: np.ndarray, mins: np.ndarray,
                    voxel_size: np.ndarray, padding: int,
                    sigma: float, target_faces: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Blur indicator + marching cubes + decimate → (verts, faces) at this σ.

    Skips the pymeshlab remeshing and Laplacian smoothing the production path
    runs — we need geometry for Euclidean argmax, not a Cholesky-clean RGD
    manifold. Cheaper and more frame-to-frame stable (no stochastic remesh).
    """
    if sigma <= 0:
        # gaussian_filter with sigma=0 is a no-op; MC on the bare indicator
        # gives the staircased hull approximation.
        blurred = grid
    else:
        blurred = gaussian_filter(grid, sigma=sigma)
    isovalue = blurred.max() * 0.5
    verts_vox, faces, _, _ = marching_cubes(blurred, level=isovalue,
                                              spacing=tuple(voxel_size))
    verts = verts_vox + mins - (padding * voxel_size)
    if len(faces) > target_faces:
        verts, faces = _decimate_mesh(verts, faces, target_faces=target_faces)
    return verts.astype(np.float64), faces.astype(np.int64)


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


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                       axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _draw_frame(axes: tuple, *, sigma: float,
                 mesh_verts: np.ndarray, mesh_edges: np.ndarray,
                 mesh_vert_rgb: np.ndarray,
                 seed_cielab: np.ndarray, out_rgb: np.ndarray,
                 dense_cielab: np.ndarray, dense_rgb_unit: np.ndarray,
                 hist_x_max: int) -> None:
    ax_3d, ax_2d, ax_hist = axes
    for ax in axes:
        ax.cla()

    palette, counts = _palette_from_per_seed_rgb(out_rgb)
    K = len(palette)
    K_eff = _effective_palette_size(counts)

    # --- panel 1: 3D gamut + isosurface wireframe + per-vertex colour ---
    ax_3d.scatter(dense_cielab[:, 1], dense_cielab[:, 2], dense_cielab[:, 0],
                  c=dense_rgb_unit, s=4, alpha=0.30, edgecolors="none")
    seg = mesh_verts[mesh_edges][:, :, [1, 2, 0]]
    lc = Line3DCollection(seg, colors="0.25", linewidths=0.4, alpha=0.30)
    ax_3d.add_collection3d(lc)
    # Subsample mesh vertices for scatter (5000 verts × scatter is slow in 3D).
    if len(mesh_verts) > 800:
        idx = np.linspace(0, len(mesh_verts) - 1, 800).astype(int)
    else:
        idx = np.arange(len(mesh_verts))
    ax_3d.scatter(mesh_verts[idx, 1], mesh_verts[idx, 2], mesh_verts[idx, 0],
                  c=mesh_vert_rgb[idx].astype(np.float32) / 255.0,
                  s=8, edgecolors="none", alpha=0.85, zorder=10)
    ax_3d.set_xlim(-100, 100); ax_3d.set_ylim(-100, 100); ax_3d.set_zlim(0, 100)
    ax_3d.set_xlabel("a*"); ax_3d.set_ylabel("b*"); ax_3d.set_zlabel("L*")
    ax_3d.view_init(elev=20, azim=-55)
    ax_3d.set_title(f"hull-Gaussian isosurface\n$\\sigma$ = {sigma:.2f}",
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--smin", type=float, default=0.0,
                   help="minimum σ (0 = bare voxelised hull, staircased)")
    p.add_argument("--smax", type=float, default=8.0,
                   help="maximum σ (large = smoothed blob)")
    p.add_argument("--target-faces", type=int, default=5000,
                   help="decimate isosurface to this many faces (matches "
                        "subdiv=2 of animate_alpha_hull for cross-comparison)")
    p.add_argument("--vox", type=int, default=256,
                   help="voxel resolution along the longest CIELAB axis")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None,
                   help="output path (default figures/gaussian_hull_animation/"
                        "gaussian_hull.gif or .mp4)")
    p.add_argument("--mp4", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out) if args.out else fig_path(
        "gaussian_hull_animation",
        "gaussian_hull.mp4" if args.mp4 else "gaussian_hull.gif",
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- precompute static data once ---
    print("[anim] sampling seed grid (interval=16)...", flush=True)
    seed_rgb_int, _ = generate_input_grid("sRGB", interval=16)
    seed_cielab = srgb_to_lab(seed_rgb_int.astype(np.float64))
    seed_rgb_u8 = seed_rgb_int.astype(np.uint8)
    all_rgbs_seed = seed_rgb_int.astype(np.int64)

    print("[anim] sampling dense gamut for 3D backdrop (interval=8)...",
          flush=True)
    dense_rgb_int, _ = generate_input_grid("sRGB", interval=8)
    rng = np.random.default_rng(0)
    if len(dense_rgb_int) > 4000:
        idx = rng.choice(len(dense_rgb_int), size=4000, replace=False)
        dense_rgb_int = dense_rgb_int[idx]
    dense_cielab_bg = srgb_to_lab(dense_rgb_int.astype(np.float64))
    dense_rgb_unit = dense_rgb_int.astype(np.float32) / 255.0

    grid, mins, voxel_size, padding = _voxelise_dense_gamut(vox=args.vox)

    # --- σ schedule ---
    sigmas = np.linspace(args.smin, args.smax, args.frames)
    print(f"[anim] {args.frames} frames, sigma {args.smin} -> {args.smax}",
          flush=True)

    # --- per-frame mesh + Euclidean argmax (the slow part) ---
    import time
    per_frame_out_rgb = []
    per_frame_mesh = []  # list of (verts, faces, edges, vert_rgb)
    ks = []
    t0 = time.perf_counter()
    for fi, sigma in enumerate(sigmas):
        verts, faces = _mesh_at_sigma(grid, mins, voxel_size, padding,
                                       float(sigma), args.target_faces)
        # Per-vertex nearest displayable sRGB.
        vert_rgb = _nearest_rgb(verts, seed_cielab, all_rgbs_seed).astype(np.uint8)
        # Euclidean argmax: for each input voxel, which mesh vertex is farthest.
        d2 = np.sum((seed_cielab[:, None, :] - verts[None, :, :]) ** 2, axis=-1)
        argmax_vert = np.argmax(d2, axis=1)
        out_rgb = vert_rgb[argmax_vert]
        per_frame_out_rgb.append(out_rgb)
        per_frame_mesh.append(
            (verts, faces, _unique_edges(faces), vert_rgb)
        )
        palette, _ = _palette_from_per_seed_rgb(out_rgb)
        ks.append(len(palette))
        if (fi + 1) % 5 == 0 or fi == 0 or fi == len(sigmas) - 1:
            dt = time.perf_counter() - t0
            rate = (fi + 1) / dt
            eta = (len(sigmas) - fi - 1) / rate if rate > 0 else float("inf")
            print(f"  [{fi + 1}/{len(sigmas)}] sigma={sigma:.3f}  "
                  f"verts={len(verts)} faces={len(faces)}  K={ks[-1]}  "
                  f"({rate:.2f} fr/s, ETA {eta / 60:.1f} min)", flush=True)
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
                  f"sigma={sigmas[frame_idx]:.3f}", flush=True)
        verts, faces, edges, vert_rgb = per_frame_mesh[frame_idx]
        _draw_frame(
            (ax_3d, ax_2d, ax_hist),
            sigma=float(sigmas[frame_idx]),
            mesh_verts=verts, mesh_edges=edges,
            mesh_vert_rgb=vert_rgb,
            seed_cielab=seed_cielab,
            out_rgb=per_frame_out_rgb[frame_idx],
            dense_cielab=dense_cielab_bg, dense_rgb_unit=dense_rgb_unit,
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
