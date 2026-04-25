"""Headless Python replacement for Unity's per-frame CEC + LUT lookup pipeline.

Mirrors ``Assets/RenderStereoBackgroundforAreaLabel.cs``. For each frame:

  1. apply label/background mask to frame -> pixels of interest
  2. 27-bin tertile CEC (see ``arlabelvis.cec``)
  3. LUT lookup at the CEC triple -> rendered label color
  4. emit one CSV row:
     ``frame,time_seconds,label_r/g/b,background_r/g/b,rendered_r/g/b``

The CSV feeds ``arlabelvis.metrics.load_and_filter`` /
``compute_gradients`` / ``print_stats``.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

CecSource = Literal["label", "background"]

import imageio.v3 as iio
import numpy as np

from arlabelvis.cec import characteristic_env_color, lookup_label_color

_log = logging.getLogger(__name__)

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


def _load_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask as (H,W) bool. Accepts grayscale / RGB / RGBA PNG.

    Priority for selecting the mask channel:
      - RGBA with a non-trivial alpha channel (multiple values): use alpha.
      - Any other multi-channel image: use the mean of the RGB channels.
      - Grayscale: use as-is.

    This handles both "alpha-encoded" masks (transparent = foreground) and
    the common case of an opaque PNG where the mask is drawn in RGB.

    If ``shape`` is given and differs from the mask's shape, a ValueError is
    raised; pass ``None`` to skip the check (callers may want to center-crop
    the frames to the mask instead).
    """
    img = np.asarray(iio.imread(path))
    if img.ndim == 3:
        if img.shape[-1] == 4 and len(np.unique(img[..., 3])) > 1:
            m = img[..., 3]
        else:
            m = img[..., :3].mean(axis=-1)
    else:
        m = img
    if shape is not None and m.shape != shape:
        raise ValueError(f"mask shape {m.shape} != frame shape {shape}")
    return m > 127


def _center_pad_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Center-pad a 2-D bool ``mask`` to ``target_hw`` with False.

    Used when the label mask was drawn at a smaller canvas than the video
    frame (e.g., Lana's mask is 3840x1920 but the original Dubai Marina
    timelapse is 3840x2160 — center-padding places the mask's label at the
    same scene location the augmented videos show, without modifying the
    frame pixels). No-op when already at target shape; raises when the
    mask is larger than the target in any dimension.
    """
    h, w = mask.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return mask
    if h > th or w > tw:
        raise ValueError(f"mask {h}x{w} larger than frame {th}x{tw}; "
                         "expected mask <= frame in both dimensions")
    out = np.zeros((th, tw), dtype=mask.dtype)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    out[y0:y0 + h, x0:x0 + w] = mask
    return out


@dataclass
class FrameStream:
    """Iterable of ``(frame_idx, frame_rgb_u8)`` pairs plus the source fps.

    Iterate directly; the ``fps`` field is the video's frame rate (read from
    container metadata for video files, defaulted to 30 for frame
    directories).
    """
    frames: "Iterable[tuple[int, np.ndarray]]"
    fps: float

    def __iter__(self):
        return iter(self.frames)


def iter_frames(video_path: str | Path) -> FrameStream:
    """Open ``video_path`` (a video file or directory of frames) and return a
    ``FrameStream`` ready to iterate."""
    path = Path(video_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
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
        return FrameStream(frames=gen(), fps=30.0)

    # imageio-ffmpeg backend. Read fps from container metadata; fall back
    # to 30 if unreadable.
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
    return FrameStream(frames=gen(), fps=fps)


def process_scene_video(
    video_path: str | Path,
    lut: np.ndarray,
    label_mask_path: str | Path,
    output_csv: str | Path,
    *,
    background_mask_path: str | Path | None = None,
    cec_source: CecSource = "label",
    num_top_bins: int = 2,
) -> int:
    """Drive the per-frame CEC + LUT-lookup loop and write `output_csv`.

    Mirrors Kwon et al.'s `RenderStereoBackgroundforAreaLabel.cs` precisely
    -- two independent masks, two CECs, the shader/script picks which feeds
    the LUT:

    * ``cec_source="label"`` (default, = Kwon's standard): CEC is computed
      from pixels under the label mask, i.e., the scene content the label
      obscures. This is the colour the rendered label must contrast with.
    * ``cec_source="background"``: CEC is computed from pixels under a
      *separate* ``background_mask_path`` (Kwon's granularity=2 variant).
      If ``background_mask_path`` is not supplied this falls back to the
      complement of the label mask, which is only occasionally what you
      want -- for a 360° equirectangular video the label is a tiny rectangle
      and the complement is ~99% of the frame, so the mode is dominated by
      whole-scene content rather than label-local environment.

    Args:
        video_path: .mp4/.mov/.webm file, or directory of frames.
        lut: (256,256,256,3) uint8 array (see ``arlabelvis.luts.lut_to_srgb_u8``).
        label_mask_path: PNG mask marking the label region (pixels >127 included).
        output_csv: where to write the per-frame CSV.
        background_mask_path: optional; if None, the background mask defaults
            to the complement of the label mask. Required in practice when
            ``cec_source="background"`` is meant to match Kwon's granularity=2.
        cec_source: "label" or "background" — which mask's CEC drives the lookup.
        num_top_bins: N in "take top-N most-populated bins" (Unity default 2).

    Returns the number of frames written.
    """
    if cec_source not in ("label", "background"):
        raise ValueError(f"cec_source must be 'label' or 'background', got {cec_source!r}")
    if cec_source == "background" and background_mask_path is None:
        import warnings
        warnings.warn(
            "cec_source='background' without background_mask_path falls back "
            "to the label-complement mask; pass an explicit background mask "
            "to match Kwon's granularity=2 CEC.",
            stacklevel=2,
        )
    stream = iter_frames(video_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0

    # Load masks at their own shape; they get center-padded with False up to
    # the video's frame shape on the first frame we see. This lets a single
    # 3840x1920 mask drive both the augmented 3840x1920 scenes and the
    # original 3840x2160 source without per-scene mask files or cropping.
    raw_label_mask = _load_mask(label_mask_path, shape=None)
    raw_bg_mask: np.ndarray | None = None
    if background_mask_path is not None:
        raw_bg_mask = _load_mask(background_mask_path, shape=None)
        if raw_bg_mask.shape != raw_label_mask.shape:
            raise ValueError(
                f"label mask {raw_label_mask.shape} != background mask "
                f"{raw_bg_mask.shape}; both must share a frame shape."
            )

    label_mask: np.ndarray | None = None
    bg_mask: np.ndarray | None = None
    fixed_shape: tuple[int, int] | None = None

    with output_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)

        for idx, frame in stream:
            if label_mask is None:
                fh_shape = frame.shape[:2]
                label_mask = _center_pad_mask(raw_label_mask, fh_shape)
                if raw_bg_mask is not None:
                    bg_mask = _center_pad_mask(raw_bg_mask, fh_shape)
                else:
                    bg_mask = ~label_mask
                fixed_shape = fh_shape
            elif frame.shape[:2] != fixed_shape:
                raise ValueError(
                    f"frame {idx} shape {frame.shape[:2]} differs from the "
                    f"first frame's shape {fixed_shape}; the masks were padded "
                    f"to that shape and won't apply to the new frame."
                )
            label_px = frame[label_mask]
            bg_px = frame[bg_mask]

            label_cec = characteristic_env_color(label_px, num_top_bins)
            bg_cec = characteristic_env_color(bg_px, num_top_bins)
            shader_in = label_cec if cec_source == "label" else bg_cec
            rendered = lookup_label_color(lut, shader_in)

            row = FrameRow(
                frame=idx, time_seconds=idx / stream.fps,
                label_rgb=label_cec, background_rgb=bg_cec, rendered_rgb=rendered,
            )
            w.writerow(row.as_csv())
            n_rows += 1

    _log.info("wrote %d rows to %s", n_rows, output_csv)
    return n_rows


# ----------------------------- CLI ----------------------------------------
