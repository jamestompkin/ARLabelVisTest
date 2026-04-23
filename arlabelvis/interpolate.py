"""Sparse -> dense 256^3 LUT interpolation with fast, vectorized I/O.

Takes sparse (sRGB index, output-value) pairs sampled on a regular grid and
upsamples to a dense 256^3 LUT. Writes either a numpy binary (.npy, fast,
~0.5 s at 256^3) or a comma-separated text file (.txt, ~30 s via np.savetxt;
needed for Unity's `LookupTableRender.cs`).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _write_lut(lut: np.ndarray, path: Path) -> None:
    """Write a dense (256,256,256,3) LUT to disk, format chosen by extension."""
    if path.suffix == ".npy":
        np.save(path, lut)
    elif path.suffix == ".txt":
        np.savetxt(path, lut.reshape(-1, 3), delimiter=",", fmt="%.6f")
    else:
        raise ValueError(
            f"Unsupported output extension {path.suffix!r}. Use .npy (fast) or .txt (Unity)."
        )
    print(f"Saved LUT -> {path}")


def _assemble_dense(rgb_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Direct scatter: place each (r,g,b)->value at lut[r,g,b].

    Assumes the inputs cover every voxel (interval == 1 case).
    """
    lut = np.zeros((256, 256, 256, 3), dtype=np.float32)
    lut[rgb_idx[:, 0], rgb_idx[:, 1], rgb_idx[:, 2]] = values
    return lut


def _assemble_sparse_and_interp(rgb_idx: np.ndarray, values: np.ndarray,
                                interval: int) -> np.ndarray:
    """Scatter onto a sparse regular grid, then trilinearly interpolate to 256^3."""
    step_values = np.arange(-1, 256, interval)
    step_values[0] = 0
    k = len(step_values)
    sparse = np.zeros((k, k, k, 3), dtype=np.float32)

    i_idx = np.searchsorted(step_values, rgb_idx[:, 0])
    j_idx = np.searchsorted(step_values, rgb_idx[:, 1])
    k_idx = np.searchsorted(step_values, rgb_idx[:, 2])
    sparse[i_idx, j_idx, k_idx] = values

    fn = RegularGridInterpolator((step_values, step_values, step_values), sparse)
    # Query all 16.7M voxels at once. ~1-2 s peak memory.
    r, g, b = np.meshgrid(np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32), indexing="ij")
    pts = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    lut = fn(pts).reshape(256, 256, 256, 3).astype(np.float32)
    return lut


def interpolate_interval(rgbVals, labVals, new_LAB_filepath: str, interval: int) -> None:
    """Upsample sparse (rgbVals, labVals) to a dense 256^3 LUT and save.

    Args:
        rgbVals: (N, 3) int array of input RGB indices sampled on a regular
            ``interval``-spaced grid (e.g., stepSize=8 -> values 0,7,15,...,255).
        labVals: (N, 3) float array of output color values (typically CIELAB).
        new_LAB_filepath: output path. ``.npy`` -> binary; ``.txt`` -> legacy
            per-voxel text (Unity-readable).
        interval: the stepSize used to generate ``rgbVals``. ``interval == 1``
            means the inputs are already dense.
    """
    rgb_idx = np.asarray(rgbVals, dtype=np.int32)
    values = np.asarray(labVals, dtype=np.float32)
    if rgb_idx.shape != values.shape:
        raise ValueError(f"row count mismatch: {rgb_idx.shape} vs {values.shape}")

    if interval == 1:
        lut = _assemble_dense(rgb_idx, values)
    else:
        lut = _assemble_sparse_and_interp(rgb_idx, values, interval)

    _write_lut(lut, Path(new_LAB_filepath))


def interpolate_files(rgbVals, labVals, new_LAB_filepath: str) -> None:
    """interval == 1 convenience alias (direct scatter, no trilinear interp)."""
    interpolate_interval(rgbVals, labVals, new_LAB_filepath, interval=1)


def interpolate_from_files(rgb_filepath: str, lab_filepath: str, new_LAB_filepath: str) -> None:
    """Read paired sparse text files and write a dense LUT.

    Each input file has one triple per line (comma-separated). rgb values are
    integer input indices; lab values are floats. The output extension picks
    the binary-vs-text path.
    """
    rgb_idx = np.loadtxt(rgb_filepath, delimiter=",", dtype=np.int32)
    values = np.loadtxt(lab_filepath, delimiter=",", dtype=np.float32)

    # Infer interval from the sparse sample density.
    ux = np.unique(rgb_idx[:, 0])
    if len(ux) == 256:
        interval = 1
    else:
        diffs = np.diff(ux)
        interval = int(diffs[diffs > 0][0]) if len(diffs) else 1

    interpolate_interval(rgb_idx, values, new_LAB_filepath, interval)
