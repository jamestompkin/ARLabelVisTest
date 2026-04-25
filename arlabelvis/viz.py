"""LUT visualization: load a cached ``(256, 256, 256, 3)`` array and render.

Three styles:

1. ``render_srgb_cube_isometric`` — three visible outer faces of the input
   sRGB cube, coloured by the LUT output.
2. ``render_color_space_pointcloud`` — 3D scatter in an intermediate colour
   space (CIELAB / OKLAB) coloured by the LUT output.
3. ``render_hue_histogram`` — 1D hue distribution of the LUT outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import pyvista as pv


# ----------------------------- I/O -----------------------------------------

# ----------------------------- renders -------------------------------------

def render_srgb_cube_isometric(
    lut: np.ndarray,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    window_size: tuple[int, int] = (1200, 1200),
    show: bool = False,
) -> None:
    """Render three visible outer faces of the 256^3 sRGB input cube.

    Default path uses PyVista/VTK for a high-quality per-cell RGB render.
    Falls back to a matplotlib ``plot_surface`` renderer if the VTK DLLs
    cannot be loaded (e.g., blocked by Windows Application Control) so the
    paper pipeline still runs end-to-end. The mpl fallback is coarser and
    visually distinct but suitable for review drafts.
    """
    try:
        _render_srgb_cube_pyvista(lut, save_path, title=title,
                                  window_size=window_size, show=show)
    except ImportError as e:
        if "vtk" not in str(e).lower():
            raise
        print(f"[viz] PyVista/VTK import failed ({e}); using matplotlib "
              f"fallback. Unblock VTK and re-run for publication-quality "
              f"renders.", flush=True)
        _render_srgb_cube_matplotlib(lut, save_path, title=title)


def _render_srgb_cube_pyvista(
    lut: np.ndarray,
    save_path,
    *,
    title=None,
    window_size=(1200, 1200),
    show=False,
) -> None:
    """PyVista/VTK cube renderer. Each face is a 256x256 uniform cell grid
    with per-cell RGB (no texture, no interpolation, no face normals in the
    color path). Faces are guaranteed to be geometrically and color-wise
    continuous across shared edges."""
    n = lut.shape[0]

    pl = pv.Plotter(off_screen=not show, window_size=window_size)
    pl.set_background("white")

    # VTK ImageData cell ordering: cell_id = x + y*nx + z*nx*ny (x fastest).
    # For each face one axis has a single cell layer; the two varying axes
    # need to land in the right slots for continuity across shared edges.
    for fixed_axis, fixed_val in [(0, n - 1), (1, n - 1), (2, n - 1)]:
        # Face slice. axis=0 -> (y,z); axis=1 -> (x,z); axis=2 -> (x,y).
        face = np.take(lut, fixed_val, axis=fixed_axis)  # (n, n, 3)

        # Flatten into VTK cell order. The two in-plane axes come out of
        # np.take in the order of the remaining original axes (skipping
        # fixed_axis). VTK wants x fastest, then y, then z. For each face
        # the "first varying axis" in `face` corresponds to the lower of the
        # two remaining 3-D axes, which is also VTK's faster axis -> so we
        # transpose so that axis becomes the fast (inner) one when flattened.
        cells = np.ascontiguousarray(
            face.transpose(1, 0, 2).reshape(-1, 3)
        ).astype(np.uint8)

        dims = [n + 1, n + 1, n + 1]
        dims[fixed_axis] = 2  # 2 points = 1 cell thick along the fixed axis
        origin = [0.0, 0.0, 0.0]
        origin[fixed_axis] = float(fixed_val)

        grid = pv.ImageData(dimensions=tuple(dims), spacing=(1.0, 1.0, 1.0),
                            origin=tuple(origin))
        grid.cell_data["rgb"] = cells
        pl.add_mesh(grid, scalars="rgb", rgb=True,
                    show_edges=False, lighting=False)

    # True isometric view of the +XYZ corner (camera along (1,1,1)/sqrt(3))
    size = float(n - 1)
    center = size / 2.0
    dist = size * 2.2
    pl.camera_position = [
        (center + dist, center + dist, center + dist),  # camera
        (center, center, center),                        # focus
        (0.0, 0.0, 1.0),                                 # up
    ]
    pl.enable_parallel_projection()  # proper orthographic isometric
    pl.camera.parallel_scale = size * 0.9
    pl.show_axes()

    if title:
        pl.add_text(title, position="upper_edge", font_size=12, color="black")

    if save_path:
        pl.show(screenshot=str(save_path))
    elif show:
        pl.show()
    else:
        pl.close()


def _render_srgb_cube_matplotlib(
    lut: np.ndarray,
    save_path,
    *,
    title=None,
    figsize=(8, 8),
    downsample: int = 4,
    dpi: int = 150,
) -> None:
    """Matplotlib fallback cube renderer. Plots 3 outer faces as
    ``plot_surface`` meshes with per-face-cell RGB colors. Downsampled by
    ``downsample`` (default 4 -> 64x64 cells per face) for rendering speed;
    the PyVista path is the reference at full 256x256."""
    n = lut.shape[0]
    step = max(1, int(downsample))
    idx = np.arange(0, n, step, dtype=np.int32)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    for axis, val in [(0, n - 1), (1, n - 1), (2, n - 1)]:
        face = np.take(lut, val, axis=axis).astype(np.float32) / 255.0
        face_ds = face[::step, ::step]
        nn = face_ds.shape[0]
        if axis == 0:
            Y, Z = np.meshgrid(idx, idx, indexing="ij")
            X = np.full_like(Y, val, dtype=float)
        elif axis == 1:
            X, Z = np.meshgrid(idx, idx, indexing="ij")
            Y = np.full_like(X, val, dtype=float)
        else:
            X, Y = np.meshgrid(idx, idx, indexing="ij")
            Z = np.full_like(X, val, dtype=float)
        ax.plot_surface(X, Y, Z, facecolors=face_ds,
                        rcount=nn, ccount=nn, shade=False, edgecolor="none")

    ax.set_xlim(0, n - 1); ax.set_ylim(0, n - 1); ax.set_zlim(0, n - 1)
    ax.view_init(elev=30, azim=45)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_color_space_pointcloud(
    lut: np.ndarray,
    points: np.ndarray,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    space_label: tuple[str, str, str] = ("L*", "a*", "b*"),
    figsize: tuple[float, float] = (7, 6),
    dpi: int = 200,
    subsample: int = 32,
    elev: float = 25,
    azim: float = 45,
    marker_size: float = 40,
    show: bool = False,
) -> plt.Figure:
    """3D scatter in the intermediate color space colored by LUT output sRGB.

    Args:
        lut: (256,256,256,3) uint8 LUT output sRGB.
        points: (N, 3) point cloud in the intermediate color space. Each point
            corresponds to an RGB triple via `idx = i // step_size^2, ...` — caller
            is responsible for matching cardinality to their LUT sampling. Typical
            usage: pass CIELAB coords of the (r,g,b) grid at the same step_size
            used to build the LUT.
        space_label: axis labels for the three intermediate-space axes.
        subsample: show every Nth point for speed / clarity.

    Matches the style of the existing cielab_rgd_*.png figures.
    """
    # Derive per-point LUT colors by grid-indexing into lut.
    # Assumes points are in bijection with a uniform RGB grid of side `step`
    # mapped onto [0, lut_dim-1].
    lut_dim = lut.shape[0]
    if lut.shape[:3] != (lut_dim, lut_dim, lut_dim):
        raise ValueError(
            f"render_color_space_pointcloud expects a cubic LUT; got shape {lut.shape}"
        )
    step = int(round(len(points) ** (1 / 3)))
    if step ** 3 != len(points):
        raise ValueError(f"points length {len(points)} is not a perfect cube")
    stride = lut_dim // step
    r = (np.arange(step) * stride).clip(0, lut_dim - 1)
    R, G, B = np.meshgrid(r, r, r, indexing="ij")
    colors = lut[R.ravel(), G.ravel(), B.ravel()] / 255.0

    P = points[::subsample]
    C = colors[::subsample]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=marker_size,
               edgecolors="none", alpha=1.0)
    ax.set_xlabel(space_label[0]); ax.set_ylabel(space_label[1]); ax.set_zlabel(space_label[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def render_hue_histogram(
    lut: np.ndarray,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    bins: int = 360,
    figsize: tuple[float, float] = (9, 3),
    dpi: int = 200,
    show: bool = False,
) -> plt.Figure:
    """Hue distribution of LUT outputs as a 1D histogram with hue-colored bars.

    Matches results/histograms/AllCandidateLABvals_*_hue.png style.
    """
    rgb = lut.reshape(-1, 3).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    hue = hsv[:, 0]  # in [0, 1)

    counts, edges = np.histogram(hue, bins=bins, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Bar colors: hue-cycle at high saturation/value so the bars are legible
    bar_colors = hsv_to_rgb(np.stack([centers, np.ones_like(centers), np.ones_like(centers)], axis=1))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(centers * 360.0, counts, width=(360.0 / bins),
           color=bar_colors, edgecolor="none")
    ax.set_xlim(0, 360)
    ax.set_xlabel("Hue (degrees)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


