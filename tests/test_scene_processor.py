"""Smoke test for scene_video.process_scene_video: build a synthetic scene +
mask + LUT, run the pipeline, and verify the CSV round-trips through metrics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import imageio.v3 as iio

from arlabelvis.cec import tertile_bins, characteristic_env_color
from arlabelvis.scene_video import process_scene_video

HERE = Path(__file__).parent
OUT = HERE / "_smoke_out"
OUT.mkdir(exist_ok=True)


def make_synth_video(path: Path, n_frames: int = 30, w: int = 256, h: int = 256):
    """A scene whose background slowly shifts across hue — introduces real CEC motion."""
    from matplotlib.colors import hsv_to_rgb
    frames = []
    for i in range(n_frames):
        hue = i / n_frames
        bg = (hsv_to_rgb(np.stack([np.full((h, w), hue),
                                   np.full((h, w), 0.8),
                                   np.full((h, w), 0.8)], axis=-1)) * 255).astype(np.uint8)
        frame = bg.copy()
        cy, cx = h // 2, w // 2
        frame[cy-30:cy+30, cx-60:cx+60] = (255, 255, 255)
        frames.append(frame)
    iio.imwrite(path, np.stack(frames, axis=0), fps=30, plugin="FFMPEG", codec="libx264")


def make_label_mask(path: Path, w: int = 256, h: int = 256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//2 - 30 : h//2 + 30, w//2 - 60 : w//2 + 60] = 255
    iio.imwrite(path, mask)


def complement_lut() -> np.ndarray:
    """Trivial 'farthest color' LUT: each input RGB -> its complement."""
    r, g, b = np.meshgrid(np.arange(256), np.arange(256), np.arange(256),
                          indexing="ij")
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return (255 - rgb).astype(np.uint8)


def unit_checks():
    # (0,0,0) -> bin 0; (255,255,255) -> bin 26; (100,100,100) -> bin 13 (1,1,1)
    px = np.array([[0, 0, 0], [255, 255, 255], [100, 100, 100], [85, 170, 255]],
                  dtype=np.uint8)
    b = tertile_bins(px)
    assert b[0] == 0, b[0]
    assert b[1] == 26, b[1]
    assert b[2] == 1*9 + 1*3 + 1, b[2]
    assert b[3] == 0*9 + 1*3 + 2, b[3]
    print("[PASS] tertile_bins")

    red_px = np.full((1000, 3), (200, 20, 30), dtype=np.uint8)
    cec = characteristic_env_color(red_px)
    assert 195 <= cec[0] <= 205 and cec[1] <= 25 and cec[2] <= 35, cec
    print(f"[PASS] CEC on uniform red: {cec}")


def main():
    print("unit checks...")
    unit_checks()

    video_path = OUT / "synth_scene.mp4"
    mask_path = OUT / "label_mask.png"
    print(f"generating synthetic 30-frame video at {video_path}...")
    make_synth_video(video_path)
    make_label_mask(mask_path)

    print("building synthetic complement LUT...")
    lut = complement_lut()

    csv_out = OUT / "label_colors_export.csv"
    print("running pipeline...")
    n = process_scene_video(video_path, lut, mask_path, csv_out,
                            cec_source="background")
    assert n == 30, f"expected 30 rows, got {n}"

    print("checking metrics.py can consume the CSV...")
    from arlabelvis.metrics import load_and_filter, compute_gradients, print_stats
    df = load_and_filter(csv_out)
    grads = compute_gradients(df)
    print_stats(df, grads)
    print("[PASS] metrics.py roundtrip")
    print(f"\ndone. outputs in {OUT}")


if __name__ == "__main__":
    main()
