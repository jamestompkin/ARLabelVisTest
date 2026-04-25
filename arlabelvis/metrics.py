"""LUT diversity and per-frame gradient metrics.

Two independent families:

- ``color_diversity`` — four scalar summaries (unique / effective palette /
  hue entropy / mean NN ΔE) of a ``(256, 256, 256, 3)`` LUT's output set.

- ``load_and_filter`` / ``compute_gradients`` / ``print_stats`` — per-frame
  RGB-gradient reduction of a scene-processor CSV (see
  ``arlabelvis.scene_video``).
"""
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv

from arlabelvis.colors import srgb_to_lab


def color_diversity(lookup_data: np.ndarray, *,
                    pixel_scale: Literal["auto", "u8", "unit"] = "auto",
                    hue_bins: int = 360,
                    nn_sample_size: int = 20_000) -> dict:
    """Four complementary diversity metrics over the LUT output set.

    Input `lookup_data` is the 256^3 LUT of output *farthest colors* (the
    sRGB or LAB value assigned to each input sRGB triple). The four metrics
    answer different questions about how varied those outputs actually are:

    unique_colors
        Number of distinct output colors in the LUT (after quantising to
        8-bit sRGB if needed). Raw count. Corner-clumping LUTs land ~6-10;
        smooth LUTs thousands.

    palette_effective_size
        Inverse-Simpson / Hill-number-of-order-2 over the output-color
        frequencies: `(sum n_i)^2 / sum n_i^2`. Weights unique count by how
        evenly each color is actually used. If one color dominates a
        nominally-1000-color LUT, this drops toward 1. Reduces to the raw
        unique count when every output color appears equally.

    hue_entropy_bits
        Shannon entropy of the output hue distribution, in bits, computed
        over `hue_bins` equal-width hue bins. Max is `log2(hue_bins)` for
        a perfectly uniform hue spread.

    mean_nn_delta_e
        Mean of per-color nearest-neighbor distances in CIELAB among
        *unique* output colors, using an L2-in-LAB approximation of
        ΔE76. Captures perceptual spread.

    `nn_sample_size` caps the unique-colors set when computing mean NN for
    performance; >20k unique colors get randomly sub-sampled before the
    KDTree query.

    `pixel_scale` controls how floats are interpreted: "u8" means
    `lookup_data` is already 0-255 (clip + cast), "unit" means it is
    0-1 (multiply by 255 first), and the default "auto" guesses based
    on the array max — fragile on dim images, but convenient for the
    common 256^3 LUT case where every corner is hit.
    """
    pixels = lookup_data.reshape(-1, 3)
    if pixels.dtype == np.uint8:
        pixels_u8 = pixels
    else:
        if pixel_scale == "auto":
            scale = "u8" if pixels.max() > 1.5 else "unit"
        else:
            scale = pixel_scale
        if scale == "u8":
            pixels_u8 = np.clip(pixels, 0, 255).astype(np.uint8)
        elif scale == "unit":
            pixels_u8 = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"unknown pixel_scale={scale!r}")

    unique, counts = np.unique(pixels_u8, axis=0, return_counts=True)
    n_unique = int(len(unique))
    n_total = int(counts.sum())

    palette_effective = float((n_total ** 2) / (counts.astype(np.int64) ** 2).sum()) \
                        if n_total > 0 else 0.0

    rgb_unique = unique.astype(np.float32) / 255.0
    hsv_unique = rgb_to_hsv(rgb_unique)
    hue = hsv_unique[:, 0]
    hist, _ = np.histogram(hue, bins=hue_bins, range=(0.0, 1.0), weights=counts)
    p = hist / max(hist.sum(), 1)
    nz = p > 0
    hue_entropy_bits = float(-(p[nz] * np.log2(p[nz])).sum())

    from scipy.spatial import cKDTree
    if n_unique >= 2:
        if n_unique > nn_sample_size:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_unique, size=nn_sample_size, replace=False)
            rgb_sample = rgb_unique[idx]
        else:
            rgb_sample = rgb_unique
        # Use the in-house ``srgb_to_lab`` so this module's perceptual
        # spread is computed against the same CIELAB the LUT pipeline
        # selected against.
        lab_sample = srgb_to_lab(rgb_sample * 255.0)
        tree = cKDTree(lab_sample)
        dists, _ = tree.query(lab_sample, k=2)
        mean_nn_delta_e = float(dists[:, 1].mean())
    else:
        mean_nn_delta_e = 0.0

    return {
        "unique_colors": n_unique,
        "palette_effective_size": palette_effective,
        "hue_entropy_bits": hue_entropy_bits,
        "mean_nn_delta_e": mean_nn_delta_e,
    }


# ---- Scene-CSV gradient reduction ----------------------------------------


def load_and_filter(csv_path):
    """Load a per-frame scene CSV and filter out rows after large frame gaps."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    df = df.drop_duplicates(subset="frame", keep="first").reset_index(drop=True)

    frame_diffs = df["frame"].diff()
    median_step = frame_diffs.median()
    lag_threshold = median_step * 10
    lagged_mask = (frame_diffs > lag_threshold) & (frame_diffs.notna())

    n_lagged = lagged_mask.sum()
    if n_lagged > 0:
        print(f"Filtered out {n_lagged} lagged rows "
              f"(frame jump > {lag_threshold:.0f} frames)")
        df = df[~lagged_mask].reset_index(drop=True)

    print(f"Loaded {len(df)} frames after filtering")
    print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
    print(f"Time range:  {df['time_seconds'].min():.3f}s to "
          f"{df['time_seconds'].max():.3f}s")
    return df


def compute_gradients(df):
    """Per-frame RGB Euclidean gradient magnitudes for each region."""
    regions = {
        "label":      ("label_r",      "label_g",      "label_b"),
        "background": ("background_r", "background_g", "background_b"),
        "rendered":   ("rendered_r",   "rendered_g",   "rendered_b"),
    }
    results = {}
    for region, (r_col, g_col, b_col) in regions.items():
        dr = df[r_col].diff()
        dg = df[g_col].diff()
        db = df[b_col].diff()
        magnitude = np.sqrt(dr ** 2 + dg ** 2 + db ** 2)
        results[region] = {"magnitude": magnitude, "dr": dr, "dg": dg, "db": db}
    return results


def print_stats(df, gradients):
    """Human-readable gradient summary + machine-readable {max, avg, max_frame}."""
    print("\n--- Gradient Statistics ---")
    for region, data in gradients.items():
        mag = data["magnitude"].dropna()
        max_frame = mag.idxmax()
        max_time  = df.loc[max_frame, "time_seconds"] if max_frame in df.index else "?"
        print(f"\n{region.upper()} color gradient (RGB Euclidean distance per frame):")
        print(f"  Max gradient:     {mag.max():.4f}  "
              f"(frame {df.loc[max_frame, 'frame']}, t={max_time:.3f}s)")
        print(f"  Average gradient: {mag.mean():.4f}")
        print(f"  Std deviation:    {mag.std():.4f}")
        print(f"  Median gradient:  {mag.median():.4f}")
    return {
        region: {
            "max":       data["magnitude"].dropna().max(),
            "avg":       data["magnitude"].dropna().mean(),
            "max_frame": data["magnitude"].dropna().idxmax(),
        }
        for region, data in gradients.items()
    }
