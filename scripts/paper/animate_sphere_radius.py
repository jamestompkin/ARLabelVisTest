"""Animate the sphere candidate set as its radius varies.

For each frame, the icosphere candidate set has a different radius
around the gamut centroid in CIELAB. The farthest-point operator picks
each input voxel's farthest icosphere vertex (L2 in CIELAB), and that
vertex is then NN-snapped to its nearest displayable sRGB. So the
chain is::

    radius --> icosphere vertex set
            --> per-input farthest vertex
            --> per-vertex nearest displayable sRGB
            --> per-input output sRGB

The animation makes the trade-off visible:

- **Large radius** (sphere encompasses the gamut): the farthest
  icosphere vertex from any input lies in the half-space opposite the
  input. Many vertices snap to the same handful of gamut-extremum
  sRGBs (the corners and edges of the displayable region). Effective
  palette small, picks high-chroma.
- **Small radius** (sphere fully interior): the icosphere is dense
  and tightly packed inside the gamut. Each vertex snaps to a
  different nearby gamut sample. Effective palette larger, but every
  pick is low-chroma (close to the centroid).

Three panels per frame:

1. 3D CIELAB scatter — gamut point cloud (faint, coloured by sRGB) +
   wireframe icosphere at the current radius. Geometric context.
2. Input partition (a*, b*) — input voxels coloured by the LUT's
   output sRGB at this radius. The "what does it do" view.
3. Pushforward (log-log) — per-rank vlines coloured by palette entry,
   K_eff dashed line.

Usage::

    uv run python -m scripts.paper.animate_sphere_radius
    uv run python -m scripts.paper.animate_sphere_radius --frames 36 --fps 8
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

from arlabelvis.colors import srgb_to_lab
from arlabelvis.meshing import generate_input_grid
from scripts.paper._paths import fig_path

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame computation.
# ---------------------------------------------------------------------------
def _icosphere_candidates(center: np.ndarray, radius: float, *,
                           subdivisions: int = 4
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(verts, edges)`` for an icosphere of the given radius."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    verts = np.asarray(mesh.vertices) + center
    return verts, np.asarray(mesh.edges_unique)


def _farthest_argmax(seed_pts: np.ndarray, candidate_verts: np.ndarray
                      ) -> np.ndarray:
    """For each row of ``seed_pts``, return the candidate index that
    maximises L2 distance. Vectorised, no chunking — both arrays fit
    comfortably (4913 x 2562 x 3 = 38 MB)."""
    diff = seed_pts[:, None, :] - candidate_verts[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    return np.argmax(d2, axis=1)


def _palette_from_per_seed_rgb(per_seed_rgb: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(palette_rgb_u8, counts)`` sorted by descending count."""
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


# ---------------------------------------------------------------------------
# Frame rendering.
# ---------------------------------------------------------------------------
def _draw_frame(axes: tuple, *, radius: float, center: np.ndarray,
                 seed_cielab: np.ndarray, seed_rgb: np.ndarray,
                 dense_cielab: np.ndarray, dense_rgb: np.ndarray,
                 candidate_subdivisions: int,
                 wire_subdivisions: int, hist_x_max: int) -> None:
    ax_3d, ax_2d, ax_hist = axes
    for ax in axes:
        ax.cla()

    # --- compute farthest-point assignment (algorithm uses dense candidate set) ---
    cand_verts, _ = _icosphere_candidates(
        center, radius, subdivisions=candidate_subdivisions,
    )
    src_tree = cKDTree(seed_cielab)
    nn_seed_for_cand = src_tree.query(cand_verts)[1]
    cand_rgb = seed_rgb[nn_seed_for_cand]
    far_idx = _farthest_argmax(seed_cielab, cand_verts)
    out_rgb = cand_rgb[far_idx]

    palette, counts = _palette_from_per_seed_rgb(out_rgb)
    K = len(palette)
    K_eff = _effective_palette_size(counts)

    # --- panel 1: 3D gamut + sphere wireframe ---
    # Gamut scatter at moderate alpha so it reads as a coloured solid.
    ax_3d.scatter(dense_cielab[:, 1], dense_cielab[:, 2], dense_cielab[:, 0],
                  c=dense_rgb, s=4, alpha=0.45, edgecolors="none")
    # Sphere wireframe at *low* subdivisions (decoupled from the algorithm's
    # candidate-set subdivisions): subdiv=1 -> 30 edges, subdiv=2 -> 120
    # edges. The user's eye reads a clean wire-globe; the algorithm still
    # uses the dense 2562-vert sphere for its picks.
    wire = trimesh.creation.icosphere(subdivisions=wire_subdivisions,
                                       radius=radius)
    wire_verts = np.asarray(wire.vertices) + center
    wire_edges = np.asarray(wire.edges_unique)
    wire_seg = wire_verts[wire_edges]
    lc = Line3DCollection(wire_seg[:, :, [1, 2, 0]], colors="0.25",
                          linewidths=0.9, alpha=0.55)
    ax_3d.add_collection3d(lc)
    # Centre marker.
    ax_3d.scatter([center[1]], [center[2]], [center[0]],
                  c="black", s=20, marker="x", zorder=10)
    ax_3d.set_xlim(-100, 100); ax_3d.set_ylim(-100, 100); ax_3d.set_zlim(0, 100)
    ax_3d.set_xlabel("a*"); ax_3d.set_ylabel("b*"); ax_3d.set_zlabel("L*")
    ax_3d.view_init(elev=20, azim=-55)
    ax_3d.set_title(f"sphere candidate set\nradius = {radius:.1f}",
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
                   colors=marker_colors, linewidth=1.0)
    ax_hist.axvline(K_eff, color="k", lw=0.8, ls="--", alpha=0.7)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(0.7, max(hist_x_max, 10))
    ax_hist.set_ylim(1e-7, 1.0)
    ax_hist.set_xlabel("palette entry (rank, log)")
    ax_hist.set_ylabel("voxel fraction (log)")
    ax_hist.set_title(f"pushforward (K = {K},  $K_{{\\mathrm{{eff}}}}$ = {K_eff:.0f})",
                     fontsize=10)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frames", type=int, default=150,
                   help="number of frames in the animation")
    p.add_argument("--rmin", type=float, default=20.0,
                   help="minimum radius (interior to the gamut)")
    p.add_argument("--rmax", type=float, default=140.0,
                   help="maximum radius (encompassing the gamut)")
    p.add_argument("--candidate-subdivisions", type=int, default=4,
                   help="icosphere subdivisions for the candidate set "
                        "(2562 verts at 4)")
    p.add_argument("--wire-subdivisions", type=int, default=3,
                   help="icosphere subdivisions for the wireframe shown in "
                        "panel 1 (subdiv=1 -> 30 edges, 2 -> 120, 3 -> 480)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None,
                   help="output path (defaults to "
                        "figures/sphere_animation/sphere_radius.gif)")
    p.add_argument("--mp4", action="store_true",
                   help="output mp4 via ffmpeg writer (smaller file, "
                        "needs ffmpeg in PATH)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out) if args.out else fig_path(
        "sphere_animation",
        "sphere_radius.mp4" if args.mp4 else "sphere_radius.gif",
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- precompute static data: seed grid + dense gamut subsample ---
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

    # Sphere centre = centroid of CIELAB-mapped gamut.
    center = dense_cielab.mean(axis=0)
    print(f"[anim] gamut centroid (L*, a*, b*) = {center.round(2)}", flush=True)

    # --- pre-pass: max K across frames so the histogram x-axis is shared ---
    radii = np.linspace(args.rmax, args.rmin, args.frames)
    print(f"[anim] {args.frames} frames, radius {args.rmax} -> {args.rmin}",
          flush=True)
    src_tree = cKDTree(seed_cielab)
    ks = []
    for radius in radii:
        cand_verts, _ = _icosphere_candidates(
            center, radius, subdivisions=args.candidate_subdivisions,
        )
        nn_seed_for_cand = src_tree.query(cand_verts)[1]
        cand_rgb = seed_rgb_u8[nn_seed_for_cand]
        far_idx = _farthest_argmax(seed_cielab, cand_verts)
        out_rgb = cand_rgb[far_idx]
        palette, _ = _palette_from_per_seed_rgb(out_rgb)
        ks.append(len(palette))
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
        if (frame_idx % 4) == 0 or frame_idx == args.frames - 1:
            print(f"  frame {frame_idx + 1}/{args.frames}  "
                  f"radius={radii[frame_idx]:.1f}", flush=True)
        _draw_frame(
            (ax_3d, ax_2d, ax_hist),
            radius=radii[frame_idx], center=center,
            seed_cielab=seed_cielab, seed_rgb=seed_rgb_u8,
            dense_cielab=dense_cielab, dense_rgb=dense_rgb_unit,
            candidate_subdivisions=args.candidate_subdivisions,
            wire_subdivisions=args.wire_subdivisions,
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
