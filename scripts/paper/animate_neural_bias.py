"""Animate the neural-bounded candidate mesh as bound_bias varies.

bound_bias is the *geometric* knob on the neural substrate: the
asymmetric-BCE class-weight ratio used during MLP training (Liu 2024
α/β flip).

- bound_bias < 0: penalty on false positives → mesh shrinks inside the
  gamut. The "inner bound" regime.
- bound_bias = 0: balanced — best-fit boundary.
- bound_bias > 0: penalty on false negatives → mesh contains the entire
  gamut plus slop. Liu 2024's default.

|bound_bias| ≈ 3 corresponds to Liu's converged class-weight ratio of 20.

Metric is Euclidean (L2 in CIELAB) so the only thing changing is the
candidate-mesh geometry. Assignment is vertex-centric (matching
``animate_alpha_hull.py`` and ``animate_gaussian_hull.py``): each mesh
vertex finds its Euclidean-farthest sibling; inputs NN-map to the mesh
and inherit their nearest vertex's argmax sRGB.

Usage::

    uv run python -m scripts.paper.animate_neural_bias --mp4
    uv run python -m scripts.paper.animate_neural_bias --frames 60 --bmin -3 --bmax 3
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import _decimate_mesh, _nearest_rgb
from arlabelvis.meshing import generate_input_grid
from arlabelvis.neural_bounding import (
    NeuralBoundingParams, neural_bounded_mesh_inprocess,
)
from scripts.paper._paths import fig_path

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


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                       axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _draw_frame(axes: tuple, *, bound_bias: float,
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

    ax_3d.scatter(dense_cielab[:, 1], dense_cielab[:, 2], dense_cielab[:, 0],
                  c=dense_rgb_unit, s=4, alpha=0.30, edgecolors="none")
    if len(mesh_edges):
        seg = mesh_verts[mesh_edges][:, :, [1, 2, 0]]
        lc = Line3DCollection(seg, colors="0.25", linewidths=0.4, alpha=0.30)
        ax_3d.add_collection3d(lc)
    if len(mesh_verts) > 800:
        idx = np.linspace(0, len(mesh_verts) - 1, 800).astype(int)
    else:
        idx = np.arange(len(mesh_verts))
    if len(idx) > 0:
        ax_3d.scatter(mesh_verts[idx, 1], mesh_verts[idx, 2], mesh_verts[idx, 0],
                      c=mesh_vert_rgb[idx].astype(np.float32) / 255.0,
                      s=8, edgecolors="none", alpha=0.85, zorder=10)
    ax_3d.set_xlim(-100, 100); ax_3d.set_ylim(-100, 100); ax_3d.set_zlim(0, 100)
    ax_3d.set_xlabel("a*"); ax_3d.set_ylabel("b*"); ax_3d.set_zlabel("L*")
    ax_3d.view_init(elev=20, azim=-55)
    bound_label = ("inner" if bound_bias < -0.05
                    else ("outer" if bound_bias > 0.05 else "balanced"))
    ax_3d.set_title(f"neural-bounded mesh ({bound_label})\n"
                    f"bound_bias = {bound_bias:+.2f}",
                    fontsize=10)

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
    p.add_argument("--bmin", type=float, default=-3.0,
                   help="minimum bound_bias (most-inner)")
    p.add_argument("--bmax", type=float, default=3.0,
                   help="maximum bound_bias (most-outer)")
    p.add_argument("--iterations", type=int, default=20000,
                   help="MLP training iterations per frame (Liu default 20k)")
    p.add_argument("--mesh-resolution", type=int, default=64,
                   help="MC grid res for fallback mesh extraction")
    p.add_argument("--target-faces", type=int, default=5000,
                   help="decimate mesh to this many faces")
    p.add_argument("--seed", type=int, default=0,
                   help="MLP RNG seed (held constant across frames so the "
                        "only thing varying is bound_bias)")
    p.add_argument("--device", default="cpu",
                   help="torch device for MLP training")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--mp4", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out) if args.out else fig_path(
        "neural_bias_animation",
        "neural_bias.mp4" if args.mp4 else "neural_bias.gif",
    )
    out.parent.mkdir(parents=True, exist_ok=True)

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

    biases = np.linspace(args.bmin, args.bmax, args.frames)
    print(f"[anim] {args.frames} frames, bound_bias {args.bmin} -> {args.bmax}",
          flush=True)

    per_frame_out_rgb: list[np.ndarray] = []
    per_frame_mesh: list = []
    ks: list[int] = []
    t0 = time.perf_counter()
    for fi, bias in enumerate(biases):
        params = NeuralBoundingParams(
            bound_bias=float(bias),
            iterations=args.iterations,
            mesh_resolution=args.mesh_resolution,
            seed=args.seed,
        )
        mesh = neural_bounded_mesh_inprocess(
            seed_cielab.astype(np.float32), params, device=args.device,
        )
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) > args.target_faces:
            verts, faces = _decimate_mesh(verts, faces,
                                            target_faces=args.target_faces)
        if len(verts) == 0:
            print(f"  [{fi + 1}/{len(biases)}] bias={bias:+.3f}  EMPTY mesh; "
                  f"recording black frame", flush=True)
            per_frame_out_rgb.append(np.zeros_like(seed_rgb_u8))
            per_frame_mesh.append((np.empty((0, 3)), np.empty((0, 3), dtype=np.int64),
                                    np.empty((0, 2), dtype=np.int64),
                                    np.empty((0, 3), dtype=np.uint8)))
            ks.append(1)
            continue
        vert_rgb = _nearest_rgb(verts, seed_cielab,
                                  all_rgbs_seed).astype(np.uint8)
        # Self-argmax: for each vertex, find its Euclidean-farthest sibling.
        # Vertex-centric assignment matches animate_alpha_hull/gaussian_hull.
        d2_self = np.sum((verts[:, None, :] - verts[None, :, :]) ** 2, axis=-1)
        vert_out_rgb = vert_rgb[np.argmax(d2_self, axis=1)]
        # Each input inherits its nearest mesh vertex's argmax sRGB.
        input_to_vert = cKDTree(verts).query(seed_cielab)[1]
        out_rgb = vert_out_rgb[input_to_vert]
        per_frame_out_rgb.append(out_rgb)
        per_frame_mesh.append(
            (verts, faces, _unique_edges(faces), vert_out_rgb)
        )
        palette, _ = _palette_from_per_seed_rgb(out_rgb)
        ks.append(len(palette))
        if (fi + 1) % 5 == 0 or fi == 0 or fi == len(biases) - 1:
            dt = time.perf_counter() - t0
            rate = (fi + 1) / dt
            eta = (len(biases) - fi - 1) / rate if rate > 0 else float("inf")
            print(f"  [{fi + 1}/{len(biases)}] bias={bias:+.3f}  "
                  f"verts={len(verts)} faces={len(faces)}  K={ks[-1]}  "
                  f"({rate:.2f} fr/s, ETA {eta / 60:.1f} min)", flush=True)
    hist_x_max = max(ks)
    print(f"[anim] K per frame = {ks}", flush=True)
    print(f"[anim] shared hist x-max = {hist_x_max}", flush=True)

    fig = plt.figure(figsize=(13.5, 4.6), dpi=96)
    ax_3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_2d = fig.add_subplot(1, 3, 2)
    ax_hist = fig.add_subplot(1, 3, 3)
    fig.tight_layout()

    def update(frame_idx):
        if (frame_idx % 10) == 0 or frame_idx == args.frames - 1:
            print(f"  render frame {frame_idx + 1}/{args.frames}  "
                  f"bias={biases[frame_idx]:+.3f}", flush=True)
        verts, faces, edges, vert_rgb = per_frame_mesh[frame_idx]
        _draw_frame(
            (ax_3d, ax_2d, ax_hist),
            bound_bias=float(biases[frame_idx]),
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
