"""Characteristic Environment Color: image-level colour-quantization primitives.

Three pure functions over RGB arrays. Not specific to videos or frames —
works on any uint8 RGB pixel array.

- ``tertile_bins(pixels)`` — per-pixel 27-bin tertile bin index.
- ``characteristic_env_color(pixels, num_top_bins)`` — Nedrich 27-bin CEC:
  mean RGB over pixels in the top-N most-populated tertile bins.
- ``lookup_label_color(lut, rgb)`` — direct ``lut[r, g, b]`` lookup.

Reference: Nedrich (2014) / Kwon et al.'s
``Assets/RenderStereoBackgroundforAreaLabel.cs`` (Unity). The tertile
thresholds (85, 170, 255 upper-bounds per channel) are Unity-compatible.
"""
from __future__ import annotations

import numpy as np


def tertile_bins(pixels_rgb: np.ndarray) -> np.ndarray:
    """Assign each ``(N, 3)`` uint8 pixel to one of 27 tertile bins.

    Unity uses upper-bound tests ``r<=85, <=170, <=255`` (per channel);
    ``np.digitize(x, [86, 171])`` gives the same tertile index 0/1/2.
    Returns an ``(N,)`` int array in ``[0, 27)``.
    """
    t = np.digitize(pixels_rgb, [86, 171]).astype(np.int32)
    return t[:, 0] * 9 + t[:, 1] * 3 + t[:, 2]


def characteristic_env_color(pixels_rgb: np.ndarray,
                             num_top_bins: int = 2) -> tuple[int, int, int]:
    """Nedrich CEC: mean RGB over pixels in the top-N most-populated tertile bins.

    Matches ``FindHistogramAverageColor`` in the Unity script with
    ``weightedBins=false`` and ``numBinsToTake=2`` (defaults).
    Returns a uint8 RGB triple suitable for direct LUT indexing.
    """
    if pixels_rgb.size == 0:
        return (0, 0, 0)
    bins = tertile_bins(pixels_rgb)
    counts = np.bincount(bins, minlength=27)
    top = np.argsort(counts)[::-1][:num_top_bins]
    mask = np.isin(bins, top)
    sel = pixels_rgb[mask]
    if sel.size == 0:
        return (0, 0, 0)
    avg = sel.mean(axis=0)
    return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])))


def lookup_label_color(lut: np.ndarray,
                       rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Trilinear-free LUT lookup: ``lut[r, g, b]`` directly (uint8 input)."""
    r, g, b = (int(np.clip(c, 0, 255)) for c in rgb)
    out = lut[r, g, b]
    return (int(out[0]), int(out[1]), int(out[2]))
