"""Animate the Gaussian-hull lookup-texture construction as mesh resolution varies.

Algorithm
---------
1. Sample sRGB values on a regular grid in [0, 255]³ (interval=16 → 4913 inputs).
2. Convert every sampled sRGB value to CIELAB.
3. Independently, voxelise the **full** sRGB gamut (interval=1, 16.7 M colours)
   in CIELAB, Gaussian-blur the binary indicator field (σ fixed), and extract
   a triangle-mesh isosurface via marching cubes.  Decimate the mesh to
   ``target_faces`` faces.  This mesh approximates the gamut boundary in CIELAB.
4. Build a lookup texture (LUT): for every input sRGB value [r, g, b], find
   the mesh vertex that is **farthest** from that input's CIELAB coordinate
   using the CIEDE2000 colour-difference metric.  Map that vertex back to its
   nearest displayable sRGB.  Store the result at LUT[r, g, b].
5. Count how many times each unique output sRGB appears in the LUT.  The
   number of distinct colours is K; their frequency distribution is the
   "pushforward".

The animation sweeps ``target_faces`` from a small value (K ≈ 4) to a large
value (K > 200), showing how mesh resolution controls the richness of the
output palette.

Three panels per frame (same layout as ``animate_alpha_hull.py``):

1. 3D CIELAB scatter — gamut backdrop + isosurface mesh wireframe + mesh
   vertices coloured by their nearest displayable sRGB.
2. Input partition (a*, b*) — input voxels coloured by their LUT output sRGB.
3. Pushforward (log-log) — K, K_eff, per-rank vlines showing the frequency
   of each output colour in the LUT.

Usage::

    uv run python -m scripts.paper.animate_gaussian_hull --mp4
    uv run python -m scripts.paper.animate_gaussian_hull --frames 60 --tf-min 4 --tf-max 5000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

# --- GPU setup ---
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _delta_e00_chunk_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between rows a (R,3) and cols b (N,3) → (R,N).

    a[:, i] is (R,) and b[:, i] is (N,). We unsqueeze to (R,1) and (1,N)
    so every intermediate is (R,N).
    """
    L1 = a[:, 0].unsqueeze(1); a1 = a[:, 1].unsqueeze(1); b1 = a[:, 2].unsqueeze(1)
    L2 = b[:, 0].unsqueeze(0); a2 = b[:, 1].unsqueeze(0); b2 = b[:, 2].unsqueeze(0)

    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - torch.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = torch.where(C1p * C2p == 0.0, torch.zeros_like(dhp), dhp)
    dHp = 2.0 * torch.sqrt(C1p * C2p) * torch.sin(torch.deg2rad(dhp / 2.0))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hdiff = torch.abs(h1p - h2p)
    hbp = torch.where(C1p * C2p == 0.0, hsum,
                      torch.where(hdiff <= 180.0, 0.5 * hsum,
                                  torch.where(hsum < 360.0, 0.5 * (hsum + 360.0),
                                              0.5 * (hsum - 360.0))))

    T = (1.0 - 0.17 * torch.cos(torch.deg2rad(hbp - 30.0))
         + 0.24 * torch.cos(torch.deg2rad(2.0 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3.0 * hbp + 6.0))
         - 0.20 * torch.cos(torch.deg2rad(4.0 * hbp - 63.0)))
    dtheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    Rc = 2.0 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    Lm50 = Lbp - 50.0
    SL = 1.0 + (0.015 * Lm50 * Lm50) / torch.sqrt(20.0 + Lm50 * Lm50)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -torch.sin(2.0 * torch.deg2rad(dtheta)) * Rc

    term_L = dLp / SL
    term_C = dCp / SC
    term_H = dHp / SH
    return torch.sqrt(term_L * term_L + term_C * term_C + term_H * term_H
                      + RT * term_C * term_H)


def _input_argmax_gpu(inputs: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """For each input (M,3), find index of farthest hull vertex (N,3) by CIEDE2000.

    Returns (M,) int64 array of indices into ``hull``.
    """
    inp = torch.as_tensor(inputs, dtype=torch.float32, device=_device)
    h = torch.as_tensor(hull, dtype=torch.float32, device=_device)
    M, N = inp.shape[0], h.shape[0]
    row_chunk = max(64, int(1.5e9 / (N * 4)))
    result = np.empty(M, dtype=np.int64)
    for start in range(0, M, row_chunk):
        end = min(start + row_chunk, M)
        dE = _delta_e00_chunk_gpu(inp[start:end], h)  # (chunk, N)
        result[start:end] = torch.argmax(dE, dim=1).cpu().numpy()
        del dE
    return result


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
    runs — we need geometry for CIEDE2000 argmax, not a Cholesky-clean RGD
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
    p.add_argument("--sigma", type=float, default=3.0,
                   help="fixed Gaussian blur σ")
    p.add_argument("--tf-min", type=int, default=4,
                   help="minimum target faces (start of sweep, K≈4)")
    p.add_argument("--tf-max", type=int, default=5000,
                   help="maximum target faces (end of sweep)")
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

    # --- target-faces schedule (log-spaced for smooth K progression) ---
    tf_schedule = np.unique(np.geomspace(args.tf_min, args.tf_max,
                                          args.frames).astype(int))
    # Recompute frames to match deduplicated schedule.
    args.frames = len(tf_schedule)
    sigma = args.sigma
    print(f"[anim] {args.frames} frames, sigma={sigma}, "
          f"target_faces {args.tf_min} -> {args.tf_max}", flush=True)

    # --- per-frame mesh + CIEDE2000 argmax (the slow part) ---
    import time
    per_frame_out_rgb = []
    per_frame_mesh = []  # list of (verts, faces, edges, vert_rgb)
    ks = []
    t0 = time.perf_counter()
    for fi, tf in enumerate(tf_schedule):
        verts, faces = _mesh_at_sigma(grid, mins, voxel_size, padding,
                                       sigma, int(tf))
        # Per-vertex nearest displayable sRGB.
        vert_rgb = _nearest_rgb(verts, seed_cielab, all_rgbs_seed).astype(np.uint8)
        # CIEDE2000 argmax: for each input voxel, which mesh vertex is farthest.
        argmax_vert = _input_argmax_gpu(seed_cielab, verts)
        out_rgb = vert_rgb[argmax_vert]
        per_frame_out_rgb.append(out_rgb)
        per_frame_mesh.append(
            (verts, faces, _unique_edges(faces), vert_rgb)
        )
        palette, _ = _palette_from_per_seed_rgb(out_rgb)
        ks.append(len(palette))
        if (fi + 1) % 5 == 0 or fi == 0 or fi == len(tf_schedule) - 1:
            dt = time.perf_counter() - t0
            rate = (fi + 1) / dt
            eta = (len(tf_schedule) - fi - 1) / rate if rate > 0 else float("inf")
            print(f"  [{fi + 1}/{len(tf_schedule)}] tf={tf}  "
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
                  f"tf={tf_schedule[frame_idx]}", flush=True)
        verts, faces, edges, vert_rgb = per_frame_mesh[frame_idx]
        _draw_frame(
            (ax_3d, ax_2d, ax_hist),
            sigma=sigma,
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
