"""Small helpers shared by figure/table scripts."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "results" / "paper" / "figures"
TAB_DIR = ROOT / "results" / "paper" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# Short-paper figures land in the Overleaf project's ieeevis2026/figures
# directory (takes precedence over ../figures/ thanks to the paper's
# \graphicspath). Overridable via $ARLABELVIS_SHORTPAPER_FIG_DIR. Default
# is James's checkout path; other contributors can set the env var.
_DEFAULT_SHORTPAPER_DIR = (
    "C:/Users/james/Brown Dropbox/James Tompkin/Apps/Overleaf/"
    "Lana Yang-Maccini Senior Thesis/ieeevis2026/figures"
)
SHORTPAPER_FIG_DIR = Path(
    os.environ.get("ARLABELVIS_SHORTPAPER_FIG_DIR", _DEFAULT_SHORTPAPER_DIR)
)


def _cielab_lut_to_u8(lut_cielab: np.ndarray) -> np.ndarray:
    """CIELAB-valued LUT → uint8 sRGB via skimage's lab2rgb."""
    from skimage.color import lab2rgb
    rgb = lab2rgb(lut_cielab.reshape(-1, 3).astype(np.float32))
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8).reshape(256, 256, 256, 3)


def _srgb_float_lut_to_u8(lut_srgb: np.ndarray) -> np.ndarray:
    """sRGB-valued (0-255 float) LUT → uint8, just a clip + cast."""
    return np.clip(lut_srgb, 0, 255).astype(np.uint8).reshape(256, 256, 256, 3)


def lut_to_srgb_u8(lut: np.ndarray, cfg) -> np.ndarray:
    """Render-ready uint8 sRGB version of a cached LUT.

    Dispatches on ``cfg.interp_space``: CIELAB-interp LUTs store continuous
    CIELAB values and go through ``lab2rgb``; sRGB-interp LUTs already store
    0-255 sRGB floats and are just cast. Use this from every figure/table
    script so callers don't have to remember which farthest-color path was
    used for a given config.
    """
    if getattr(cfg, "interp_space", "sRGB") == "CIELAB":
        return _cielab_lut_to_u8(lut)
    return _srgb_float_lut_to_u8(lut)
