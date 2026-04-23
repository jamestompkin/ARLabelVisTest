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

def save_lut(lut: np.ndarray, path: str | Path) -> None:
    """Save a dense (256,256,256,3) LUT to disk.

    Format is chosen by file extension. Dtype is preserved (float32 for
    LAB/OKLAB values from the interpolate stage; uint8 for already-rendered
    sRGB LUTs).

      - ``.npy``  numpy binary (~0.5 s save, ~0.1 s load at 256^3). Preferred
                  for the Python research loop.
      - ``.txt``  one ``r,g,b`` row per voxel in x-major order (Unity
                  compat; ~30 s at 256^3 via np.savetxt).
    """
    path = Path(path)
    lut = np.asarray(lut)
    if path.suffix == ".npy":
        np.save(path, lut)
    elif path.suffix == ".txt":
        fmt = "%d" if np.issubdtype(lut.dtype, np.integer) else "%.6f"
        np.savetxt(path, lut.reshape(-1, 3), delimiter=",", fmt=fmt)
    else:
        raise ValueError(f"unsupported extension {path.suffix!r}; use .npy or .txt")


def load_lut(lab_file: str | Path, rgb_file: str | Path | None = None,
             value: str = "lab") -> np.ndarray:
    """Load a LUT.

    Two modes, selected by what's passed:

      - ``load_lut("lut.npy")``   Fast path: loads a dense (256,256,256,3)
                                   uint8 array written by ``save_lut``. The
                                   array is returned unchanged.
      - ``load_lut(lab_file, rgb_file, value=...)``   Sparse text pair (legacy
                                   pipeline output). Reconstructs a dense
                                   256^3 array via per-axis nearest-neighbor
                                   fill of the sparse grid, with colour-space
                                   conversion per ``value``.

    ``value`` is only consulted for the sparse-text path:
      - ``"lab"``    ``lab_file`` is CIELAB; converted via ``lab2rgb``.
      - ``"oklab"``  ``lab_file`` is OKLAB.
      - ``"rgb"``    ``lab_file`` already holds 0-255 sRGB.

    Returns (256, 256, 256, 3) uint8 sRGB.
    """
    path = Path(lab_file)
    if rgb_file is None:
        if path.suffix == ".npy":
            return np.load(path)
        raise ValueError(
            f"load_lut({path!r}) without rgb_file requires a .npy path; got {path.suffix!r}"
        )

    rgb_file = Path(rgb_file)
    vals = np.loadtxt(path, delimiter=",", dtype=np.float32)
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

    # Fast path: if the input grid is a regular 3-D rectangular grid (the
    # common case — the pipeline always samples the RGB cube on a grid), we
    # can expand to 256^3 by per-axis nearest-index lookup in ~10 ms instead
    # of paying for a full scipy.ndimage.distance_transform_edt on the 256^3
    # sparse array (~8 s).
    ux = np.unique(idx[:, 0])
    uy = np.unique(idx[:, 1])
    uz = np.unique(idx[:, 2])
    if len(idx) == len(ux) * len(uy) * len(uz):
        # Scatter into a (k_x, k_y, k_z, 3) dense small array
        ix = np.searchsorted(ux, idx[:, 0])
        iy = np.searchsorted(uy, idx[:, 1])
        iz = np.searchsorted(uz, idx[:, 2])
        sparse = np.zeros((len(ux), len(uy), len(uz), 3), dtype=np.uint8)
        sparse[ix, iy, iz] = out_rgb
        # Per-axis nearest-index map from full-range [0,255] to sparse-index
        q = np.arange(256)
        nx = np.abs(q[:, None] - ux[None, :]).argmin(axis=1).astype(np.int32)
        ny = np.abs(q[:, None] - uy[None, :]).argmin(axis=1).astype(np.int32)
        nz = np.abs(q[:, None] - uz[None, :]).argmin(axis=1).astype(np.int32)
        return sparse[nx[:, None, None], ny[None, :, None], nz[None, None, :]]

    # Fallback: irregular sparse pattern, use Euclidean distance transform
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
    window_size: tuple[int, int] = (1200, 1200),
    show: bool = False,
) -> None:
    """Render three visible outer faces of the 256^3 RGB cube via PyVista/VTK.

    Each face is a 256x256 uniform cell grid with per-cell RGB (no texture,
    no interpolation, no face normals in the color path). Faces are guaranteed
    to be geometrically and color-wise continuous across shared edges: the
    cell at `lut[n-1, n-1, z]` appears identically on the R=n-1 face (at G=n-1
    edge) and on the G=n-1 face (at R=n-1 edge).
    """
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
