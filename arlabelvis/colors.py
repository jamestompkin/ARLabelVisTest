"""Colour-space conversions.

A standalone "RGB" isn't a colour space — it's a channel arrangement. Every
function in this module takes specifically *sRGB* (gamma-encoded, 0-255 or
0-1, D65 white point) as input and says so in its name. The names
``srgb_to_lab`` / ``srgb_to_oklab`` / ``srgb_to_linear_rgb`` / ``linear_rgb_to_xyz``
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


# Standard sRGB→XYZ matrix (D65). Rows are XYZ basis from the linear-RGB
# primaries, so for row-vector inputs use ``rgb @ _SRGB_TO_XYZ.T``.
_SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805],
                         [0.2126, 0.7152, 0.0722],
                         [0.0193, 0.1192, 0.9505]])
_XYZ_TO_SRGB = np.array([[ 3.2406, -1.5372, -0.4986],
                         [-0.9689,  1.8758,  0.0415],
                         [ 0.0557, -0.2040,  1.0570]])
_D65_WHITE_XYZ = np.array([95.047, 100.0, 108.883])


def linear_rgb_to_xyz(linear_rgb):
    """Linear RGB (sRGB primaries, D65) → CIE XYZ via the standard 3x3 matrix."""
    return np.asarray(linear_rgb) @ _SRGB_TO_XYZ.T


def srgb_to_oklab(srgb):
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


def oklab_to_srgb(oklab):
    """OKLAB → sRGB (gamma-encoded, returned in [0, 1]).

    Inverse of ``srgb_to_oklab``. Values outside the sRGB gamut come back
    outside [0, 1]; callers that need a displayable image should clip.
    """
    L = oklab[:, 0]
    a = oklab[:, 1]
    b = oklab[:, 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    lin_r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    lin_g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    lin_b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = np.stack([lin_r, lin_g, lin_b], axis=1)

    # Linear RGB → sRGB (gamma)
    out = np.empty_like(lin)
    mask = lin > 0.0031308
    out[mask] = 1.055 * np.power(np.clip(lin[mask], 0, None), 1 / 2.4) - 0.055
    out[~mask] = 12.92 * lin[~mask]
    return out


def xyz_to_cielab(xyz):
    """CIE XYZ (scaled s.t. D65 white = 100) → CIELAB (L* in 0-100, a*/b* signed)."""
    xyz = np.asarray(xyz, dtype=np.float64) / _D65_WHITE_XYZ
    mask = xyz > 0.008856
    f = np.where(mask, np.cbrt(np.clip(xyz, 0, None)),
                 7.787 * xyz + 16 / 116)
    L = (116 * f[:, 1]) - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def srgb_to_lab(srgb):
    """sRGB (0-255 or 0-1) → CIELAB (D65). Matches skimage.color.rgb2lab on
    legal sRGB input to within floating-point tolerance."""
    srgb = np.array(srgb, dtype=np.float64) / 255.0
    mask = srgb > 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
    srgb[~mask] /= 12.92
    # Linear RGB → XYZ (D65), in 0-100 scale so xyz_to_cielab's normalisation works.
    xyz = (srgb * 100.0) @ _SRGB_TO_XYZ.T
    return xyz_to_cielab(xyz)


def lab_to_srgb(lab):
    """CIELAB (D65) → sRGB (gamma-encoded, returned in [0, 1]).

    Inverse of ``srgb_to_lab``. Values outside the sRGB gamut come back
    outside [0, 1]; callers that need a displayable image should clip.
    """
    lab = np.asarray(lab, dtype=np.float64)
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t):
        t3 = t ** 3
        return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=1) * _D65_WHITE_XYZ
    # XYZ (0-100) → linear sRGB (0-1).
    lin = (xyz @ _XYZ_TO_SRGB.T) / 100.0

    out = np.empty_like(lin)
    mask = lin > 0.0031308
    out[mask] = 1.055 * np.power(np.clip(lin[mask], 0, None), 1 / 2.4) - 0.055
    out[~mask] = 12.92 * lin[~mask]
    return out


def convert_color(points: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
    """Convert ``(N, 3)`` points between any two of {sRGB, CIELAB, OKLAB}.

    sRGB values are 0-255 (uint8 or float). CIELAB has L*∈[0,100], a*/b*
    signed. OKLAB has L∈[0,1], a/b roughly in [-0.4, 0.4]. Round-trips
    through linear sRGB for the *LAB pairs.
    """
    if from_space == to_space:
        return np.asarray(points, dtype=np.float64).copy()
    pts = np.asarray(points, dtype=np.float64)
    # Convert to sRGB (0-255) as the pivot.
    if from_space == "sRGB":
        srgb = pts.copy()
    elif from_space == "CIELAB":
        srgb = lab_to_srgb(pts) * 255.0
    elif from_space == "OKLAB":
        srgb = oklab_to_srgb(pts) * 255.0
    else:
        raise ValueError(f"unknown from_space={from_space!r}")
    if to_space == "sRGB":
        return srgb
    if to_space == "CIELAB":
        return srgb_to_lab(srgb)
    if to_space == "OKLAB":
        return srgb_to_oklab(srgb)
    raise ValueError(f"unknown to_space={to_space!r}")


def sRGBtoOKLCH(srgb):
    """sRGB → OKLCH (polar form of OKLAB)."""
    lab = srgb_to_lab(srgb)
    l = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    c = np.sqrt(a ** 2 + b ** 2)
    h = np.arctan2(b, a) * (180 / math.pi)
    h[h < 0] += 360
    return np.stack([l / 100, c / 100, h], axis=1)


