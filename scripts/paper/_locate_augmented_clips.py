"""Correlate augmented Dubai clips against the original Dubai Marina timelapse.

For each augmented scene (``dubai_changing_bgcolors``, ``dubai_raining``) this
script finds:
  * which time window of the original 264s day-to-night timelapse the
    augmented 15s clip was taken from, AND
  * which vertical crop offset (0 / 120 / 240 px) maps the 3840x1920
    augmented frame onto the 3840x2160 original.

Output is a JSON summary + a printed table. With these offsets known we can
(a) record them in the scene manifests deterministically and (b) regenerate
high-quality augmented clips by slicing the original and re-applying Lana's
augmentations at full quality rather than using YouTube-compressed copies.

Method: compare frames under the label region only (city-scale pixels,
less likely to be touched by a sky-recolor augmentation) using MSE on the
8-bit sRGB channels. First pass: coarse scan of the original every 30
frames (1 s). Second pass: refine around the best match at frame accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = REPO_ROOT / "data/scenes/dubai_marina_timelapse/video.mp4"
MASK_PNG = REPO_ROOT / "data/scenes/dubai_changing_bgcolors/label_mask.png"
AUGMENTED = {
    "dubai_changing_bgcolors":
        REPO_ROOT / "data/scenes/dubai_changing_bgcolors/video.mp4",
    "dubai_raining":
        REPO_ROOT / "data/scenes/dubai_raining/video.mp4",
}
VERTICAL_OFFSETS = (0, 120, 240)   # the three reasonable center/top/bottom crops
REPORT = REPO_ROOT / "data/scenes/_clip_locations.json"


def _label_bbox(mask_path: Path) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) bbox of the label mask (any >127 channel)."""
    from PIL import Image
    with Image.open(mask_path) as img:
        arr = np.asarray(img)
    fg = arr[..., :3].mean(axis=-1) > 127 if arr.ndim == 3 else arr > 127
    ys, xs = np.where(fg)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _iter_original_every(step: int):
    """Yield (frame_idx, frame) from the original every ``step`` frames."""
    for i, frame in enumerate(iio.imiter(str(ORIGINAL), plugin="FFMPEG")):
        if i % step == 0:
            yield i, frame


def _frame_at(idx: int) -> np.ndarray:
    """Fetch a specific frame by iterating (ffmpeg-based imageio lacks
    efficient random access, but we only call this a handful of times)."""
    for i, frame in enumerate(iio.imiter(str(ORIGINAL), plugin="FFMPEG")):
        if i == idx:
            return frame
    raise IndexError(idx)


def _crop_bbox(frame: np.ndarray, y_off: int, bbox: tuple[int, int, int, int]
               ) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    sub = frame[y_off:y_off + 1920]
    return sub[y0:y1, x0:x1]


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(np.int32) - b.astype(np.int32)) ** 2).mean())


def _locate(aug_path: Path, bbox: tuple[int, int, int, int]) -> dict:
    aug_frame0 = next(iio.imiter(str(aug_path), plugin="FFMPEG"))
    if aug_frame0.shape[:2] != (1920, 3840):
        raise ValueError(f"expected aug 1920x3840, got {aug_frame0.shape}")
    y0, y1, x0, x1 = bbox
    aug_patch = aug_frame0[y0:y1, x0:x1]

    # Coarse: scan original every 30 frames (~1s at 29.97fps).
    best = (-1, -1, float("inf"))     # (frame_idx, y_off, mse)
    for i, frame in _iter_original_every(step=30):
        for y_off in VERTICAL_OFFSETS:
            cand = _crop_bbox(frame, y_off, bbox)
            score = _mse(aug_patch, cand)
            if score < best[2]:
                best = (i, y_off, score)
    print(f"  coarse best: frame {best[0]} @ y_off={best[1]}  mse={best[2]:.1f}")

    # Refine: search +/- 30 frames at frame accuracy, at the best y_off.
    lo = max(0, best[0] - 30)
    hi = best[0] + 30
    y_off = best[1]
    best2 = (best[0], y_off, best[2])
    for i, frame in enumerate(iio.imiter(str(ORIGINAL), plugin="FFMPEG")):
        if i < lo: continue
        if i > hi: break
        cand = _crop_bbox(frame, y_off, bbox)
        score = _mse(aug_patch, cand)
        if score < best2[2]:
            best2 = (i, y_off, score)
    print(f"  refined    : frame {best2[0]} @ y_off={best2[1]}  mse={best2[2]:.1f}")
    return {
        "start_frame": best2[0],
        "start_time_s": round(best2[0] / 29.97, 3),
        "vertical_offset_px": best2[1],
        "label_bbox_mse": round(best2[2], 2),
    }


def main():
    bbox = _label_bbox(MASK_PNG)
    print(f"label bbox (y0,y1,x0,x1): {bbox}  "
          f"({bbox[1]-bbox[0]}x{bbox[3]-bbox[2]} px)")

    out: dict = {
        "original": str(ORIGINAL.relative_to(REPO_ROOT)).replace("\\", "/"),
        "mask": str(MASK_PNG.relative_to(REPO_ROOT)).replace("\\", "/"),
        "label_bbox_yxyx": bbox,
        "results": {},
    }
    for name, path in AUGMENTED.items():
        print(f"\n=== {name} ===")
        out["results"][name] = _locate(path, bbox)

    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT.relative_to(REPO_ROOT)}")
    for name, r in out["results"].items():
        print(f"  {name}: start frame {r['start_frame']} "
              f"({r['start_time_s']:.2f}s), y_off={r['vertical_offset_px']} px")


if __name__ == "__main__":
    main()
