"""Regression test for ``arlabelvis.colors``.

Background: until 2026-04-25 the sRGB→XYZ matrix in ``srgb_to_lab``
was applied with the wrong orientation (``srgb @ M`` rather than
``srgb @ M.T``), so white came out at L*≈107 instead of 100, red at
(66, 24, 32) instead of (53, 80, 67), etc. — every CIELAB-working-
space LUT in the paper was numerically wrong. This test pins the
correct values so the matrix can't silently regress.

Run with ``uv run python -m tests.test_colors``.
"""
from __future__ import annotations

import numpy as np

from arlabelvis.colors import (convert_color, lab_to_srgb, oklab_to_srgb,
                                srgb_to_lab, srgb_to_oklab)


def _close(a, b, atol=0.1, label=""):
    """Assert max abs diff <= atol; print and raise on failure."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b).max()
    if diff > atol:
        raise AssertionError(
            f"{label}: max abs diff {diff:.6f} > {atol}\n  ours: {a}\n  ref:  {b}"
        )
    print(f"  ok  {label}: max abs diff {diff:.4e}")


def test_canonical_cielab_anchors():
    """Pin the sRGB primaries / corners to standard CIELAB values.

    These are the textbook anchors (CIE D65, sRGB primaries, 8-bit
    quantisation). Off-by-five-percent on white is the symptom of a
    matrix orientation regression, so atol is tight.
    """
    print("test_canonical_cielab_anchors")
    cases = [
        ([0, 0, 0],       [  0.00,  0.00,  0.00]),
        ([255, 255, 255], [100.00,  0.00,  0.00]),
        ([255, 0, 0],     [ 53.24, 80.09, 67.20]),
        ([0, 255, 0],     [ 87.74, -86.18, 83.18]),
        ([0, 0, 255],     [ 32.30, 79.20, -107.86]),
        ([128, 128, 128], [ 53.59,  0.00,  0.00]),
    ]
    for srgb, ref_lab in cases:
        ours = srgb_to_lab(np.array([srgb], dtype=np.float64))[0]
        _close(ours, ref_lab, atol=0.05, label=f"srgb_to_lab({srgb})")


def test_lab_round_trip():
    """Random sRGB triples should round-trip through CIELAB to within ~0.1 byte."""
    print("test_lab_round_trip")
    rng = np.random.default_rng(0)
    srgb = rng.integers(0, 256, size=(5000, 3)).astype(np.float64)
    lab = srgb_to_lab(srgb)
    back = lab_to_srgb(lab) * 255.0
    _close(back, srgb, atol=0.5, label="srgb -> lab -> srgb")


def test_oklab_round_trip():
    """OKLAB round-trip: should be near-exact (no quantisation in OKLAB)."""
    print("test_oklab_round_trip")
    rng = np.random.default_rng(1)
    srgb = rng.integers(0, 256, size=(5000, 3)).astype(np.float64)
    ok = srgb_to_oklab(srgb)
    back = oklab_to_srgb(ok) * 255.0
    _close(back, srgb, atol=0.05, label="srgb -> oklab -> srgb")


def test_convert_color_pivot():
    """``convert_color`` should agree with the direct conversion functions."""
    print("test_convert_color_pivot")
    rng = np.random.default_rng(2)
    srgb = rng.integers(0, 256, size=(2000, 3)).astype(np.float64)
    lab_direct = srgb_to_lab(srgb)
    lab_via = convert_color(srgb, "sRGB", "CIELAB")
    _close(lab_via, lab_direct, atol=0.05, label="convert sRGB->CIELAB")
    ok_direct = srgb_to_oklab(srgb)
    ok_via = convert_color(srgb, "sRGB", "OKLAB")
    _close(ok_via, ok_direct, atol=1e-6, label="convert sRGB->OKLAB")


def test_skimage_agreement():
    """``srgb_to_lab`` should match ``skimage.color.rgb2lab`` to within
    ~0.05 L* (residual from the truncated 4-decimal matrix coefficients)."""
    print("test_skimage_agreement")
    try:
        from skimage.color import rgb2lab
    except ImportError:
        print("  skipped (skimage not installed)")
        return
    rng = np.random.default_rng(3)
    srgb = rng.integers(0, 256, size=(5000, 3)).astype(np.float64)
    ours = srgb_to_lab(srgb)
    ref = rgb2lab(srgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    _close(ours, ref, atol=0.05, label="srgb_to_lab vs skimage rgb2lab")


def main():
    test_canonical_cielab_anchors()
    test_lab_round_trip()
    test_oklab_round_trip()
    test_convert_color_pivot()
    test_skimage_agreement()
    print("\nALL OK")


if __name__ == "__main__":
    main()
