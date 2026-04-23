"""Colour-space conversions.

A standalone "RGB" isn't a colour space — it's a channel arrangement. Every
function in this module takes specifically *sRGB* (gamma-encoded, 0-255 or
0-1, D65 white point) as input and says so in its name. The names
``sRGBtoLAB`` / ``sRGBtoOKLAB`` / ``srgb_to_linear_rgb`` / ``linear_rgb_to_xyz``
make that explicit.
"""
import numpy as np
import math


def srgb_to_linear_rgb(srgb):
    """sRGB (gamma-encoded, 0-255 or 0-1) → linear RGB (0-1).

    Accepts uint8 or float inputs; always returns float in [0, 1].
    """
    srgb = np.array(srgb) / 255.0
    mask = srgb > 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
    srgb[~mask] /= 12.92
    return srgb


def linear_rgb_to_xyz(linear_rgb):
    """Linear RGB (sRGB primaries, D65) → CIE XYZ via the standard 3x3 matrix."""
    return np.dot(linear_rgb, np.array([[0.4124, 0.3576, 0.1805],
                                        [0.2126, 0.7152, 0.0722],
                                        [0.0193, 0.1192, 0.9505]]))


def sRGBtoOKLAB(srgb):
    """sRGB (0-255 or 0-1) → OKLAB (Ottosson 2020)."""
    lin = srgb_to_linear_rgb(srgb)

    l = 0.4122214708 * lin[:, 0] + 0.5363325363 * lin[:, 1] + 0.0514459929 * lin[:, 2]
    m = 0.2119034982 * lin[:, 0] + 0.6806995451 * lin[:, 1] + 0.1073969566 * lin[:, 2]
    s = 0.0883024619 * lin[:, 0] + 0.2817188376 * lin[:, 1] + 0.6299787005 * lin[:, 2]

    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    lin[:, 0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    lin[:, 1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    lin[:, 2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lin


def xyz_to_cielab(xyz):
    """CIE XYZ (scaled s.t. D65 white = 100) → CIELAB (L* in 0-100, a*/b* signed)."""
    xyz /= np.array([95.047, 100.0, 108.883])
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1 / 3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

    L = (116 * xyz[:, 1]) - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])
    return np.stack([L, a, b], axis=1)


def sRGBtoLAB(srgb):
    """sRGB (0-255 or 0-1) → CIELAB (D65). Matches skimage.color.rgb2lab on
    legal sRGB input to within floating-point tolerance."""
    srgb = np.array(srgb) / 255.0
    mask = srgb > 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
    srgb[~mask] /= 12.92
    srgb *= 100

    xyz = np.dot(srgb, np.array([[0.4124, 0.3576, 0.1805],
                                 [0.2126, 0.7152, 0.0722],
                                 [0.0193, 0.1192, 0.9505]]))

    xyz /= np.array([95.047, 100.0, 108.883])
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1 / 3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16 / 116)

    L = (116 * xyz[:, 1]) - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])
    return np.stack([L, a, b], axis=1)


def sRGBtoOKLCH(srgb):
    """sRGB → OKLCH (polar form of OKLAB)."""
    lab = sRGBtoLAB(srgb)
    l = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    c = np.sqrt(a ** 2 + b ** 2)
    h = np.arctan2(b, a) * (180 / math.pi)
    h[h < 0] += 360
    return np.stack([l / 100, c / 100, h], axis=1)


