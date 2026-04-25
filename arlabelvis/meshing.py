"""Boundary-mesh construction from a 3-D point cloud + the dense input grid.

Three exports:

- ``points_to_mesh``       Gaussian-smoothed gamut mesh (voxelise → marching
                           cubes → laplacian smooth → decimate → remesh).
                           Consumed by the gaussian-smoothing path in
                           ``arlabelvis.luts.build_candidates``.
- ``generate_labs``        Dense sRGB input grid + its CIELAB image. The
                           pre-input_space-generalisation entry point;
                           kept as a thin wrapper over ``generate_input_grid``
                           for back-compat.
- ``generate_input_grid``  Sample a regular grid in any of {sRGB, CIELAB,
                           OKLAB}. Returns ``(input_pts, srgb_pts)``: the
                           grid in input-space coordinates plus their sRGB
                           equivalents (the latter is what the dense LUT is
                           indexed by, for downstream consumption).
"""
import logging

import numpy as np
import trimesh
import trimesh.smoothing
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

from arlabelvis.colors import (convert_color, lab_to_srgb, oklab_to_srgb,
                                srgb_to_lab)

_log = logging.getLogger(__name__)


def points_to_mesh(all_labs, sigma=0.25, vox=256, pre_decimate_smooth: int = 3,
                   post_decimate_smooth: int = 5, target_faces=50000):
    """Voxelise a LAB point cloud and marching-cubes out a watertight boundary mesh."""
    _log.info("points_to_mesh: %d input points", len(all_labs))

    mins = all_labs.min(axis=0)
    maxs = all_labs.max(axis=0)
    ranges = maxs - mins + 1e-12

    longest = ranges.max()
    per_axis_res = np.round((ranges / longest) * vox).astype(int)
    per_axis_res = np.clip(per_axis_res, 16, vox)

    padding = 2
    scale = (per_axis_res - 1 - 2 * padding) / ranges

    idx = np.floor((all_labs - mins) * scale).astype(np.int32) + padding
    idx = np.clip(idx, 0, per_axis_res - 1)

    grid = np.zeros(tuple(per_axis_res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    blurred = gaussian_filter(grid, sigma=sigma)

    isovalue = blurred.max() * 0.5
    voxel_size = ranges / (per_axis_res - 1)

    verts_vox, faces, normals, _ = marching_cubes(blurred, level=isovalue,
                                                   spacing=tuple(voxel_size))

    verts = verts_vox + mins - (padding * voxel_size)

    tm = trimesh.Trimesh(vertices=verts, faces=faces,
                         vertex_normals=normals, process=True)

    if pre_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=pre_decimate_smooth)

    if target_faces and len(tm.faces) > target_faces:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=tm.vertices.astype(np.float64),
                               face_matrix=tm.faces.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5, targetlen=pymeshlab.PercentageValue(0.8))
    m = ms.current_mesh()
    tm = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(),
                         process=True)

    if post_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=post_decimate_smooth)

    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)

    components = trimesh.graph.connected_components(tm.edges)
    if len(components) > 1:
        _log.warning("%d connected components; keeping the largest", len(components))
        tm = tm.submesh([max(components, key=len)], append=True)

    _log.info("points_to_mesh: %d verts, %d faces, watertight=%s",
              len(tm.vertices), len(tm.faces), tm.is_watertight)
    return tm


def generate_labs(interval: int = 16):
    """Generate the input sRGB grid (at ``interval``) and its CIELAB image.

    Equivalent to ``generate_input_grid('sRGB', interval)`` followed by a
    sRGB→CIELAB conversion. Kept for back-compat — new code should use
    ``generate_input_grid`` directly.
    """
    all_rgbs = np.array([[r - 1, g - 1, b - 1] for r in range(0, 257, interval)
                                              for g in range(0, 257, interval)
                                              for b in range(0, 257, interval)])
    all_rgbs = np.where(all_rgbs < 0, 0, all_rgbs)
    all_rgbs = np.where(all_rgbs > 255, 255, all_rgbs)
    all_labs = srgb_to_lab(all_rgbs)
    return all_rgbs, all_labs


# Per-space sampling ranges for ``generate_input_grid``. CIELAB / OKLAB
# extents are the empirical bounding box of the *sRGB gamut* in those
# spaces (measured by transforming a 200k-sample uniform sRGB scatter,
# rounded out a touch); sampling outside these is wasteful since
# ``generate_input_grid`` already drops out-of-gamut points downstream.
# CIELAB: L*∈[0,100], a*∈[-86,98], b*∈[-108,94].
# OKLAB:  L∈[0,1],   a∈[-0.234,0.276], b∈[-0.311,0.198].
_INPUT_SPACE_RANGES = {
    "sRGB":   (np.array([0.0, 0.0, 0.0]),       np.array([255.0, 255.0, 255.0])),
    "CIELAB": (np.array([0.0, -90.0, -110.0]),  np.array([100.0, 100.0, 95.0])),
    "OKLAB":  (np.array([0.0, -0.24, -0.32]),   np.array([1.0, 0.28, 0.20])),
}


def generate_input_grid(input_space: str, interval: int = 16
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Sample a regular grid in ``input_space``; return ``(input_pts, srgb_pts)``.

    ``interval`` is interpreted as the *sRGB-cube* stride (so ``interval=1``
    is dense 256^3, ``interval=8`` is 33^3, etc.); the same number of
    samples per axis is used for non-sRGB spaces but laid out across the
    space's natural range (see ``_INPUT_SPACE_RANGES``) rather than 0-255.

    Sample-count formula: ``n_per_axis = ceil(256 / interval) + 1`` for
    sRGB (matching ``range(0, 257, interval)``); same count is reused
    verbatim for non-sRGB spaces so the grid resolutions are comparable.

    For ``input_space='sRGB'`` this matches ``generate_labs(interval)``
    exactly: ``input_pts == srgb_pts`` and both are integer-valued in
    ``[0, 255]``.

    For ``input_space='CIELAB'`` / ``'OKLAB'``: the returned ``srgb_pts``
    is *float* and is the conversion of each grid point into sRGB. Some
    grid cells fall outside the displayable sRGB gamut so their
    ``srgb_pts`` values land outside ``[0, 255]``; callers that need only
    in-gamut points should mask on
    ``np.all((srgb_pts >= 0) & (srgb_pts <= 255), axis=1)``.
    """
    if input_space not in _INPUT_SPACE_RANGES:
        raise ValueError(f"unknown input_space={input_space!r}")

    if input_space == "sRGB":
        # Match ``generate_labs`` exactly: integer sRGB values, with the
        # ``r - 1`` shift that avoids the 256-aliased boundary, then clipped.
        # This is a quirk of the original sampling — preserved for cache
        # compatibility with prior LUTs.
        all_rgbs = np.array([[r - 1, g - 1, b - 1]
                             for r in range(0, 257, interval)
                             for g in range(0, 257, interval)
                             for b in range(0, 257, interval)])
        all_rgbs = np.clip(all_rgbs, 0, 255)
        return all_rgbs, all_rgbs.astype(np.float64)

    n_per_axis = len(range(0, 257, interval))
    lo, hi = _INPUT_SPACE_RANGES[input_space]
    axis = np.linspace(0.0, 1.0, n_per_axis)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    input_pts = lo[None, :] + grid * (hi - lo)[None, :]
    srgb_pts = convert_color(input_pts, input_space, "sRGB")
    return input_pts, srgb_pts
