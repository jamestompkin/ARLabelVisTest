"""Headless Python replacement for Unity's per-frame CEC + LUT lookup pipeline.

Mirrors `Assets/RenderStereoBackgroundforAreaLabel.cs`:

  for each frame:
    1. apply label/background mask to frame -> pixels of interest
    2. 27-bin tertile histogram (Nedrich binning; bin upper-bounds = 85, 170, 255)
    3. take top-N most-populated bins, average their pixels -> CEC
    4. LUT[CEC.r, CEC.g, CEC.b] -> rendered label color
    5. write one CSV row: frame,time_seconds,label_r/g/b,background_r/g/b,rendered_r/g/b

Consumer: `utils/metrics.py::process_final_colors` reads this CSV schema directly.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import csv
import numpy as np
import imageio.v3 as iio

from arlabelvis.viz import load_lut


# ----------------------------- core ops -----------------------------------

def tertile_bins(pixels_rgb: np.ndarray) -> np.ndarray:
    """Assign each (N,3) uint8 pixel to one of 27 tertile bins.

    Unity uses upper-bound tests r<=85, <=170, <=255 (per channel).
    `np.digitize(x, [86, 171])` gives the same tertile index 0/1/2.
    Returns an (N,) int array in [0, 27).
    """
    t = np.digitize(pixels_rgb, [86, 171]).astype(np.int32)  # (N,3), each channel in {0,1,2}
    return t[:, 0] * 9 + t[:, 1] * 3 + t[:, 2]


def characteristic_env_color(pixels_rgb: np.ndarray, num_top_bins: int = 2) -> tuple[int, int, int]:
    """Nedrich 27-bin CEC: mean RGB over pixels in the top-N most-populated tertile bins.

    Matches `FindHistogramAverageColor` in the Unity script with `weightedBins=false`
    and `numBinsToTake=2` (defaults).

    Returns a uint8 RGB triple (suitable for direct LUT indexing).
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


def lookup_label_color(lut: np.ndarray, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Trilinear-free LUT lookup: `lut[r,g,b]` directly (rounded uint8 input)."""
    r, g, b = (np.clip(c, 0, 255) for c in rgb)
    out = lut[r, g, b]
    return (int(out[0]), int(out[1]), int(out[2]))


# ----------------------------- pipeline -----------------------------------

@dataclass
class FrameRow:
    frame: int
    time_seconds: float
    label_rgb: tuple[int, int, int]
    background_rgb: tuple[int, int, int]
    rendered_rgb: tuple[int, int, int]

    def as_csv(self) -> list:
        return [
            self.frame, f"{self.time_seconds:.6f}",
            *self.label_rgb, *self.background_rgb, *self.rendered_rgb,
        ]


CSV_HEADER = [
    "frame", "time_seconds",
    "label_r", "label_g", "label_b",
    "background_r", "background_g", "background_b",
    "rendered_r", "rendered_g", "rendered_b",
]


def _load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a mask as (H,W) bool. Accepts grayscale / RGB / RGBA PNG.

    Priority for selecting the mask channel:
      - RGBA with a non-trivial alpha channel (multiple values): use alpha.
      - Any other multi-channel image: use the mean of the RGB channels.
      - Grayscale: use as-is.

    This handles both "alpha-encoded" masks (transparent = foreground) and
    the common case of an opaque PNG where the mask is drawn in RGB.
    """
    img = np.asarray(iio.imread(path))
    if img.ndim == 3:
        if img.shape[-1] == 4 and len(np.unique(img[..., 3])) > 1:
            m = img[..., 3]
        else:
            m = img[..., :3].mean(axis=-1)
    else:
        m = img
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} != frame shape {shape}")
    return m > 127


def iter_frames(video_path: str | Path) -> tuple[object, float]:
    """Yield (frame_idx, frame_rgb_u8) and the video fps as a side-channel."""
    path = Path(video_path)
    if path.is_dir():
        # Directory of PNG/JPG frames (sorted lexicographically)
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            raise FileNotFoundError(f"no frames in {path}")
        def gen():
            for i, f in enumerate(files):
                img = iio.imread(f)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 4:
                    img = img[..., :3]
                yield i, img.astype(np.uint8)
        return gen(), 30.0  # assume 30 fps for frame dirs
    else:
        # imageio-ffmpeg backend (installed via pyproject). Read fps from
        # container metadata; fall back to 30 if unreadable.
        try:
            meta = iio.immeta(path, plugin="FFMPEG")
            fps = float(meta.get("fps", 30.0))
        except Exception:
            fps = 30.0

        def gen():
            for i, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[..., :3]
                yield i, frame.astype(np.uint8)
        return gen(), fps


def process_scene_video(
    video_path: str | Path,
    lut: np.ndarray,
    label_mask_path: str | Path,
    output_csv: str | Path,
    *,
    background_mask_path: str | Path | None = None,
    cec_source: str = "label",
    num_top_bins: int = 2,
) -> int:
    """Drive the per-frame CEC + LUT-lookup loop and write `output_csv`.

    Args:
        video_path: .mp4/.mov/.webm file, or directory of frames.
        lut: (256,256,256,3) uint8 array, already loaded via viz.lut_render.load_lut.
        label_mask_path: PNG mask marking the label region (pixels >127 included).
        output_csv: where to write the per-frame CSV.
        background_mask_path: optional; if None, background := NOT label.
        cec_source: "label" or "background" — which mask's CEC drives the lookup.
        num_top_bins: N in "take top-N most-populated bins" (Unity default 2).

    Returns the number of frames written.
    """
    frames, fps = iter_frames(video_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    label_mask: np.ndarray | None = None
    bg_mask: np.ndarray | None = None

    with output_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)

        for idx, frame in frames:
            if label_mask is None:
                label_mask = _load_mask(label_mask_path, frame.shape[:2])
                if background_mask_path is not None:
                    bg_mask = _load_mask(background_mask_path, frame.shape[:2])
                else:
                    bg_mask = ~label_mask

            label_px = frame[label_mask]
            bg_px = frame[bg_mask]

            label_cec = characteristic_env_color(label_px, num_top_bins)
            bg_cec = characteristic_env_color(bg_px, num_top_bins)
            shader_in = label_cec if cec_source == "label" else bg_cec
            rendered = lookup_label_color(lut, shader_in)

            row = FrameRow(
                frame=idx, time_seconds=idx / fps,
                label_rgb=label_cec, background_rgb=bg_cec, rendered_rgb=rendered,
            )
            w.writerow(row.as_csv())
            n_rows += 1

    print(f"wrote {n_rows} rows to {output_csv}")
    return n_rows


# ----------------------------- CLI ----------------------------------------
