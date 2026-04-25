"""Visualise the in-process MLP-bounded gamut against the ground-truth
gamut point cloud, sweeping ``bound_bias`` from inner to outer.

Each panel shows:
  - the gamut point cloud as a translucent gray scatter (ground truth)
  - the extracted MLP mesh as a wireframe, edges coloured by signed
    distance to the gamut hull (green = inside, red = outside)
  - a side legend with the BoundReport: % verts inside, mean/max outside
    distance, fraction of gamut not enclosed by the mesh, face count

Camera and axes are fixed across panels so the geometry comparison is
honest. Output: one PNG per bias plus a combined grid.

Usage:
  uv run python -m scripts.paper.fig_neural_bound_inspect
  uv run python -m scripts.paper.fig_neural_bound_inspect --working-space OKLAB
  uv run python -m scripts.paper.fig_neural_bound_inspect --biases -3 -1 0 +1 +3
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from arlabelvis.meshing import generate_input_grid
from arlabelvis.colors import srgb_to_oklab
from arlabelvis.luts import to_working_space
from arlabelvis.neural_bounding import (
    NeuralBoundingParams,
    BoundReport,
    evaluate_bound,
    neural_bounded_mesh_inprocess,
    signed_hull_distance,
)
from scripts.paper._paths import fig_path

_log = logging.getLogger(__name__)


def _to_working(working_space: str, all_rgbs: np.ndarray) -> np.ndarray:
    """Thin wrapper kept for the inspector's interface; delegates to the
    canonical pipeline conversion."""
    return to_working_space(working_space, "sRGB", all_rgbs)


# Green-white-red colormap: cool inside, warm outside.
_INOUT_CMAP = LinearSegmentedColormap.from_list(
    "inout", ["#1a8c3a", "#bbe5be", "#f5f5f5", "#f4b9b9", "#c7311c"]
)


def _render_panel(ax, gamut_pts: np.ndarray, mesh_verts: np.ndarray,
                  mesh_faces: np.ndarray, sd: np.ndarray,
                  axis_labels: tuple[str, str, str], title: str,
                  bbox_lo: np.ndarray, bbox_hi: np.ndarray,
                  ground_truth_subsample: int = 4000) -> None:
    """One 3D panel: gamut cloud + mesh wireframe coloured by signed distance."""
    rng = np.random.default_rng(0)
    if len(gamut_pts) > ground_truth_subsample:
        idx = rng.choice(len(gamut_pts), size=ground_truth_subsample, replace=False)
        gamut_show = gamut_pts[idx]
    else:
        gamut_show = gamut_pts

    # Gamut as translucent gray scatter.
    ax.scatter(gamut_show[:, 0], gamut_show[:, 1], gamut_show[:, 2],
               c="lightgray", s=2, alpha=0.25, edgecolors="none")

    # Mesh edges coloured by signed-distance of midpoints (vertex sd averaged).
    if len(mesh_faces) > 0:
        # Build edge list from faces
        edges = np.concatenate([
            mesh_faces[:, [0, 1]],
            mesh_faces[:, [1, 2]],
            mesh_faces[:, [2, 0]],
        ], axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        # Subsample edges for rendering speed at high mesh resolutions
        if len(edges) > 2000:
            edges = edges[rng.choice(len(edges), size=2000, replace=False)]
        seg_starts = mesh_verts[edges[:, 0]]
        seg_ends = mesh_verts[edges[:, 1]]
        segments = np.stack([seg_starts, seg_ends], axis=1)
        seg_sd = 0.5 * (sd[edges[:, 0]] + sd[edges[:, 1]])
        # Symmetric color scale around 0
        scale = max(1e-6, np.abs(sd).max())
        norm = plt.Normalize(vmin=-scale, vmax=scale)
        colors = _INOUT_CMAP(norm(seg_sd))
        lc = Line3DCollection(segments, colors=colors, linewidths=0.8)
        ax.add_collection3d(lc)

    ax.set_xlim(bbox_lo[0], bbox_hi[0])
    ax.set_ylim(bbox_lo[1], bbox_hi[1])
    ax.set_zlim(bbox_lo[2], bbox_hi[2])
    ax.set_xlabel(axis_labels[0]); ax.set_ylabel(axis_labels[1]); ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=20, azim=-55)
    ax.set_title(title, fontsize=9)


def _report_text(report: BoundReport) -> str:
    return (
        f"verts: {report.n_vertices:>5d}   faces: {report.n_faces:>5d}\n"
        f"gamut coverage:        {report.gamut_coverage:>6.1%}\n"
        f"mean vert-to-gamut:    {report.mean_vert_distance_to_gamut:>6.2f}\n"
        f"max  vert-to-gamut:    {report.max_vert_distance_to_gamut:>6.2f}\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--working-space", default="CIELAB",
                   choices=["CIELAB", "OKLAB", "sRGB"])
    p.add_argument("--interval", type=int, default=8,
                   help="sRGB grid stride for the gamut point cloud (default 8 → 33^3 points)")
    p.add_argument("--biases", type=float, nargs="+",
                   default=[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    p.add_argument("--iterations", type=int, default=20_000)
    p.add_argument("--mesh-resolution", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"=== gamut sampling: {args.working_space}, interval={args.interval} ===")
    _, srgb_pts = generate_input_grid("sRGB", interval=args.interval)
    all_rgbs = srgb_pts.astype(np.int64)
    gamut = _to_working(args.working_space, all_rgbs).astype(np.float32)
    print(f"    {len(gamut):,} gamut points  bbox=[{gamut.min(0)}, {gamut.max(0)}]")

    axis_labels = {
        "CIELAB": ("L*", "a*", "b*"),
        "OKLAB":  ("L",  "a",  "b"),
        "sRGB":   ("R",  "G",  "B"),
    }[args.working_space]

    # Shared bbox across panels for honest comparison
    pad = 0.10 * (gamut.max(0) - gamut.min(0))
    bbox_lo = gamut.min(0) - pad
    bbox_hi = gamut.max(0) + pad

    # Train + extract per bias
    results = []
    for bias in args.biases:
        print(f"\n=== bound_bias = {bias:+.2f} ===")
        params = NeuralBoundingParams(
            bound_bias=bias,
            iterations=args.iterations,
            mesh_resolution=args.mesh_resolution,
            seed=args.seed,
        )
        mesh = neural_bounded_mesh_inprocess(gamut, params)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if len(verts) == 0:
            print(f"    EMPTY MESH at bias={bias:+.2f} — level set didn't cross threshold")
            results.append((bias, verts, faces, np.array([]), None))
            continue
        sd = signed_hull_distance(verts, gamut)
        report = evaluate_bound(mesh, gamut)
        print(f"    {report}")
        results.append((bias, verts, faces, sd, report))

    # Per-bias single-panel PNGs
    out_dir_label = "neural_bound_inspect"
    for bias, verts, faces, sd, report in results:
        fig = plt.figure(figsize=(7, 6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        bias_tag = f"bias{bias:+.1f}".replace("+", "p").replace("-", "n").replace(".", "o")
        title = (
            f"{args.working_space}  bound_bias={bias:+.2f}\n"
            f"({'inner' if bias < 0 else 'balanced' if bias == 0 else 'outer'} bound)"
        )
        _render_panel(ax, gamut, verts, faces, sd, axis_labels, title,
                      bbox_lo, bbox_hi)
        if report is not None:
            ax.text2D(0.02, 0.02, _report_text(report), transform=ax.transAxes,
                      fontsize=7, family="monospace", verticalalignment="bottom",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        fig.tight_layout()
        out = fig_path(out_dir_label, f"{args.working_space}_{bias_tag}.png")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"  wrote {out.name}")

    # Combined grid
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4.0 * cols, 3.4 * rows), dpi=96)
    for i, (bias, verts, faces, sd, report) in enumerate(results):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        title = f"bias={bias:+.2f}"
        _render_panel(ax, gamut, verts, faces, sd, axis_labels, title,
                      bbox_lo, bbox_hi)
        if report is not None:
            ax.text2D(0.02, 0.02,
                      f"cov:{report.gamut_coverage:>5.1%}  "
                      f"d:{report.mean_vert_distance_to_gamut:>4.1f}",
                      transform=ax.transAxes, fontsize=7, family="monospace",
                      verticalalignment="bottom",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    fig.suptitle(
        f"MLP gamut-bounding sweep — {args.working_space}\n"
        f"green edges = inside gamut hull,  red edges = outside",
        fontsize=11,
    )
    fig.tight_layout()
    out = fig_path(out_dir_label, f"{args.working_space}_sweep.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
