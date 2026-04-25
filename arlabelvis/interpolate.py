"""Sparse -> dense 256^3 LUT assembly.

The pipeline picks a *farthest-color* output for each seed in a sparse
sRGB grid. The dense 256^3 LUT for runtime use is filled by
nearest-neighbour assignment from those seeds — every dense voxel
inherits whichever seed's algorithmic pick is closest in sRGB. This
preserves the discrete partition the farthest-point operator
produced; trilinear blending between two unrelated picks would output
sRGB triplets that aren't farthest from anything, violating the
method's semantics.

A trilinear path remains available via ``method='linear'`` for callers
that explicitly want smoothing (e.g. AR shaders that prefer
continuous transitions over partition-faithful jumps), but it is no
longer the default.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def dense_lut_from_sparse(rgb_vals: np.ndarray, lut_vals: np.ndarray, *,
                          interval: int,
                          regular_grid: bool = True,
                          method: Literal["nearest", "linear"] = "nearest"
                          ) -> np.ndarray:
    """Upsample regular-grid ``(rgb_vals, lut_vals)`` to a dense 256^3 LUT.

    Args:
        rgb_vals: ``(N, 3)`` int sRGB indices on a regular ``interval``-
            spaced grid (e.g., ``interval=8`` → values 0, 7, 15, …, 255).
        lut_vals: ``(N, 3)`` float output values (sRGB or CIELAB depending on
            the caller's output_space).
        interval: the stride used to generate ``rgb_vals``. ``interval == 1``
            means the inputs already densely cover every voxel and the
            direct-scatter fast path is used; otherwise the
            ``method``-driven fill is used.
        regular_grid: must be True (kept for backward compatibility).
            Scattered seeds (non-sRGB ``input_space``) live in
            ``arlabelvis.luts._scattered_dense_lut``.
        method:
          - ``'nearest'`` (default) — every dense voxel takes its nearest
            seed's algorithmic pick. Preserves the farthest-point
            partition exactly. The dense LUT contains exactly the K
            unique algorithmic outputs, never their blends.
          - ``'linear'`` — trilinear interpolation between adjacent seeds.
            Produces smooth transitions in the output cube but yields
            sRGB triplets that aren't algorithmic picks. Kept for
            callers that explicitly want shader-smooth behaviour.

    Returns a ``(256, 256, 256, 3) float32`` array.
    """
    if not regular_grid:
        raise ValueError(
            "dense_lut_from_sparse(regular_grid=False) is no longer supported; "
            "call arlabelvis.luts._scattered_dense_lut directly for non-sRGB "
            "input grids — it does NN in working space, not sRGB."
        )
    rgb_idx = np.asarray(rgb_vals, dtype=np.int32)
    values = np.asarray(lut_vals, dtype=np.float32)
    if rgb_idx.shape != values.shape:
        raise ValueError(f"row count mismatch: {rgb_idx.shape} vs {values.shape}")

    if interval == 1:
        return _assemble_dense(rgb_idx, values)
    if method == "nearest":
        return _assemble_sparse_and_nearest(rgb_idx, values, interval)
    if method == "linear":
        return _assemble_sparse_and_linear(rgb_idx, values, interval)
    raise ValueError(f"unknown method={method!r}")


# Back-compat: old callers pass a destination path; keep the file-based entry
# point as a thin wrapper. Will be removed once no one imports it by this name.
def interpolate_interval(rgb_vals, lut_vals, output_path: str, interval: int) -> None:
    from pathlib import Path
    path = Path(output_path)
    if path.suffix != ".npy":
        raise ValueError(f"output must be a .npy path; got {path.suffix!r}")
    lut = dense_lut_from_sparse(rgb_vals, lut_vals, interval=interval)
    np.save(path, lut)


def _assemble_dense(rgb_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Direct scatter: place each (r,g,b) -> value at ``lut[r,g,b]``."""
    lut = np.zeros((256, 256, 256, 3), dtype=np.float32)
    lut[rgb_idx[:, 0], rgb_idx[:, 1], rgb_idx[:, 2]] = values
    return lut


def _assemble_sparse_and_nearest(rgb_idx: np.ndarray, values: np.ndarray,
                                  interval: int) -> np.ndarray:
    """Each dense voxel takes its nearest seed's value.

    Implementation: for each axis independently, snap every dense
    coordinate 0..255 to the nearest seed coordinate, then use the
    snapped triplet to index the sparse lattice. Voronoi cells along
    each axis are half-intervals; ties at midpoints round consistently
    via numpy's argmin behaviour. This is faster than a 3-D KDTree
    query because the seed lattice is axis-aligned.
    """
    step_values = np.arange(-1, 256, interval)
    step_values[0] = 0
    if step_values[-1] != 255:
        step_values = np.append(step_values, 255)
    k = len(step_values)
    sparse = np.zeros((k, k, k, 3), dtype=np.float32)

    i_idx = np.searchsorted(step_values, rgb_idx[:, 0])
    j_idx = np.searchsorted(step_values, rgb_idx[:, 1])
    k_idx = np.searchsorted(step_values, rgb_idx[:, 2])
    sparse[i_idx, j_idx, k_idx] = values

    # Snap each dense coordinate to its nearest seed coordinate per axis.
    coords = np.arange(256)
    # Nearest-seed index for every dense coordinate.
    nearest_idx = np.abs(coords[:, None] - step_values[None, :]).argmin(axis=1)

    R = nearest_idx[:, None, None]
    G = nearest_idx[None, :, None]
    B = nearest_idx[None, None, :]
    R = np.broadcast_to(R, (256, 256, 256))
    G = np.broadcast_to(G, (256, 256, 256))
    B = np.broadcast_to(B, (256, 256, 256))
    return sparse[R, G, B].astype(np.float32)


def _assemble_sparse_and_linear(rgb_idx: np.ndarray, values: np.ndarray,
                                 interval: int) -> np.ndarray:
    """Scatter onto a sparse regular grid, then *trilinearly* interpolate.

    Kept available behind ``method='linear'`` for callers that want
    shader-smooth output transitions. Note: produces sRGB triplets that
    are not algorithmic picks; do not use this when the LUT is being
    analysed for palette diversity, K_eff, etc.
    """
    step_values = np.arange(-1, 256, interval)
    step_values[0] = 0
    k = len(step_values)
    sparse = np.zeros((k, k, k, 3), dtype=np.float32)

    i_idx = np.searchsorted(step_values, rgb_idx[:, 0])
    j_idx = np.searchsorted(step_values, rgb_idx[:, 1])
    k_idx = np.searchsorted(step_values, rgb_idx[:, 2])
    sparse[i_idx, j_idx, k_idx] = values

    fn = RegularGridInterpolator((step_values, step_values, step_values), sparse)
    r, g, b = np.meshgrid(np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32), indexing="ij")
    pts = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    return fn(pts).reshape(256, 256, 256, 3).astype(np.float32)
