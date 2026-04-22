"""Lookup-table visualization for the thesis/short-paper figures.

Reads the paired text format produced by main.py's INTERPOLATE stage
(AllCandidateLABvals_*.txt + AllCorrespondingRGBVals_*.txt), builds a
dense (256,256,256,3) array of LUT output RGB, and renders three styles
of figure:

1. `render_rgb_cube_isometric` — 3 visible outer faces of the 256^3 input-RGB
   cube, colored by the LUT output. The "cube" visualization in the paper.
2. `render_color_space_pointcloud` — 3D scatter of the intermediate color-space
   points (CIELAB / Oklab / RGB) colored by the LUT output. Matches the style
   of Lana's existing figures (cielab_rgd_05.png, oklab_rgd_05.png, etc).
3. `render_hue_histogram` — 1D hue distribution of the LUT outputs. Matches
   the `..._hue.png` figures in results/histograms/.

The loader is forgiving: if the RGB-index file does not densely cover 256^3
(e.g. stepSize > 1 before interpolation), unfilled voxels are nearest-neighbor
filled so downstream rendering has no holes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt
from skimage.color import lab2rgb
import pyvista as pv


# ----------------------------- I/O -----------------------------------------

def load_lut(lab_file: str | Path, rgb_file: str | Path, value: str = "lab") -> np.ndarray:
    """Load a LUT from the codebase's paired text format.

    Args:
        lab_file: one comma-separated LAB (or RGB) triple per line.
        rgb_file: one comma-separated *input* RGB index triple per line.
        value: "lab" (default) if lab_file is CIELAB; "oklab" or "rgb" for
            those intermediate spaces. The return is always sRGB.

    Returns:
        (256, 256, 256, 3) uint8 array of LUT output sRGB values. Unfilled
        voxels are nearest-neighbor filled.
    """
    lab_file, rgb_file = Path(lab_file), Path(rgb_file)
    vals = np.loadtxt(lab_file, delimiter=",", dtype=np.float32)
    idx = np.loadtxt(rgb_file, delimiter=",", dtype=np.int32)
    if vals.shape != idx.shape:
        raise ValueError(f"row count mismatch: {vals.shape} vs {idx.shape}")

    if value == "lab":
        out_rgb = np.clip(lab2rgb(vals) * 255.0, 0, 255).astype(np.uint8)
    elif value == "oklab":
        out_rgb = np.clip(_oklab_to_rgb(vals) * 255.0, 0, 255).astype(np.uint8)
    elif value == "rgb":
        out_rgb = np.clip(vals, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown value={value!r}")

    lut = np.zeros((256, 256, 256, 3), dtype=np.uint8)
    filled = np.zeros((256, 256, 256), dtype=bool)
    lut[idx[:, 0], idx[:, 1], idx[:, 2]] = out_rgb
    filled[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    if not filled.all():
        _, nearest = distance_transform_edt(~filled, return_distances=True, return_indices=True)
        lut = lut[nearest[0], nearest[1], nearest[2]]
    return lut


def _oklab_to_rgb(ok: np.ndarray) -> np.ndarray:
    """Inverse of utils/color_spaces.RGBtoOKLAB (vectorized, returns [0,1] sRGB)."""
    L, a, b = ok[:, 0], ok[:, 1], ok[:, 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3; m = m_ ** 3; s = s_ ** 3
    lin_r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    lin_g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    lin_b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = np.stack([lin_r, lin_g, lin_b], axis=1)
    # linear sRGB -> sRGB
    a_thr = 0.0031308
    return np.where(lin <= a_thr, 12.92 * lin, 1.055 * np.power(np.clip(lin, 0, None), 1/2.4) - 0.055)


# ----------------------------- renders -------------------------------------

def render_rgb_cube_isometric(
    lut: np.ndarray,
    save_path: str | Path | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
    dpi: int = 200,
    stride: int = 2,
    elev: float = 25,
    azim: float = 30,
    show: bool = False,
) -> plt.Figure:
    """Render three visible outer faces of the 256^3 RGB cube, textured with
    LUT output colors. Paper-ready 3D isometric view.

    stride downsamples the 256x256 faces for faster rendering (stride=2 -> 128x128).
    """
    n = lut.shape[0]
    s = slice(None, None, stride)
    # Three visible faces when viewed from +elev/+azim: r=n-1, g=n-1, b=n-1
    # Use quads with per-face facecolors.
    def face(indices_r, indices_g, indices_b):
        """Make a (H, W, 3) quad grid + colors for the given face."""
        R, G, B = np.meshgrid(indices_r, indices_g, indices_b, indexing="ij")
        shape = R.shape
        colors = lut[R, G, B] / 255.0
        return R.reshape(-1), G.reshape(-1), B.reshape(-1), colors.reshape(-1, 3), shape

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Plot three faces as surface plots
    idx = np.arange(0, n, stride, dtype=np.int32)
    # Face r=n-1 (ri fixed high, vary G and B)
    G, B = np.meshgrid(idx, idx, indexing="ij")
    R = np.full_like(G, n - 1)
    ax.plot_surface(R, G, B, rstride=1, cstride=1,
                    facecolors=lut[n - 1][s, s] / 255.0, shade=False,
                    antialiased=False, linewidth=0)
    # Face g=n-1
    R, B = np.meshgrid(idx, idx, indexing="ij")
    G = np.full_like(R, n - 1)
    ax.plot_surface(R, G, B, rstride=1, cstride=1,
                    facecolors=lut[:, n - 1][s, s] / 255.0, shade=False,
                    antialiased=False, linewidth=0)
    # Face b=n-1
    R, G = np.meshgrid(idx, idx, indexing="ij")
    B = np.full_like(R, n - 1)
    ax.plot_surface(R, G, B, rstride=1, cstride=1,
                    facecolors=lut[:, :, n - 1][s, s] / 255.0, shade=False,
                    antialiased=False, linewidth=0)

    ax.set_xlabel("R"); ax.set_ylabel("G"); ax.set_zlabel("B")
    ax.set_xlim(0, n - 1); ax.set_ylim(0, n - 1); ax.set_zlim(0, n - 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    return fig


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
            corresponds to an RGB triple via `idx = i // stepSize^2, ...` — caller
            is responsible for matching cardinality to their LUT sampling. Typical
            usage: pass CIELAB coords of the (r,g,b) grid at the same stepSize
            used to build the LUT.
        space_label: axis labels for the three intermediate-space axes.
        subsample: show every Nth point for speed / clarity.

    Matches the style of the existing cielab_rgd_*.png figures.
    """
    # Derive per-point LUT colors by grid-indexing into lut
    # Assume points are in bijection with a uniform RGB grid of side `step`.
    step = int(round(len(points) ** (1 / 3)))
    if step ** 3 != len(points):
        raise ValueError(f"points length {len(points)} is not a perfect cube")
    stride = 256 // step
    r = (np.arange(step) * stride).clip(0, 255)
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
    return fig


# ----------------------------- CLI -----------------------------------------


# ============================================================================
# Migrated from colors.py (3D pointcloud) and distances.py (mesh + geodesic field)
# ============================================================================

def plot_lab_points_3d(newPoints, RGBs=None, subsample=1):
    """
    allLABs: (N,3) LAB points
    allRGBs: (N,3) RGB points in [0,255]
    furthestRGBs: (M,3) optional RGB points to highlight
    subsample: plot every k-th point for speed
    """

    newPoints = np.asarray(newPoints)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if RGBs is not None:
        ax.scatter(
            newPoints[::subsample, 0],
            newPoints[::subsample, 1],
            newPoints[::subsample, 2],
            c=RGBs[::subsample]/255.0,
            s=120,
            linewidth=1.5
        )
    else:
        ax.scatter(
            newPoints[::subsample, 0],
            newPoints[::subsample, 1],
            newPoints[::subsample, 2],
            s=120,
            linewidth=1.5
        )

    ax.set_xlabel("L*")
    ax.set_ylabel("a*")
    ax.set_zlabel("b*")

    ax.set_title("New space (colored by original RGB)")
    ax.legend()
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.show()


def plot_geodesic_field_pyvista(V, F,
                                distances,
                                source_idx=None,
                                furthest_idx=None,
                                cmap="plasma"):
    # Convert faces to PyVista format
    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3), F]
    ).astype(np.int32)
    faces_pv = faces_pv.ravel()

    poly = pv.PolyData(V, faces_pv)

    # Attach scalar field
    poly["Geodesic Distance"] = distances

    # Normalize for consistent coloring
    clim = [float(distances.min()), float(distances.max())]

    plotter = pv.Plotter()
    plotter.add_mesh(
        poly,
        scalars="Geodesic Distance",
        cmap=cmap,
        clim=clim,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": "Regularized Geodesic Distance",
        }
    )

    if source_idx is not None:
        center = V[source_idx]
        radius = 0.02 * np.linalg.norm(V.max(0) - V.min(0))

        source_sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(
            source_sphere,
            color="lime",
            specular=1.0,
            smooth_shading=True,
        )

    # ---- Add furthest sphere ----
    if furthest_idx is not None:
        center = V[furthest_idx]
        radius = 0.02 * np.linalg.norm(V.max(0) - V.min(0))

        furthest_sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(
            furthest_sphere,
            color="red",
            specular=1.0,
            smooth_shading=True,
        )

    plotter.add_axes()
    plotter.show()


def show_original_mesh_pyvista(mesh,
                               show_edges=True,
                               show_normals=False):

    V = mesh.vertices
    F = mesh.faces

    # Convert faces to PyVista format
    faces_pv = np.hstack(
        [np.full((F.shape[0], 1), 3), F]
    ).astype(np.int32).ravel()

    poly = pv.PolyData(V, faces_pv)

    # Clean mesh (optional but useful for diagnostics)
    poly_clean = poly.clean(tolerance=1e-12)

    print("---- Mesh Diagnostics ----")
    print("Vertices:", poly.n_points)
    print("Faces:", poly.n_cells)
    print("Is manifold:", poly.is_manifold)
    print("Is all triangles:", poly.is_all_triangles)
    print("Has open edges:", poly.n_open_edges > 0)
    print("--------------------------")

    plotter = pv.Plotter()

    # Add mesh with edge overlay
    plotter.add_mesh(
        poly_clean,
        color="lightgray",
        show_edges=show_edges,
        edge_color="black",
        smooth_shading=True,
        backface_params=dict(color="orange"),
    )

    # Optional: show normals
    if show_normals:
        normals = poly_clean.compute_normals(
            cell_normals=False,
            point_normals=True,
            auto_orient_normals=False
        )

        arrows = normals.glyph(
            orient="Normals",
            scale=False,
            factor=0.05 * np.linalg.norm(V.max(0) - V.min(0))
        )

        plotter.add_mesh(arrows, color="blue")

    plotter.add_axes()
    plotter.show()
