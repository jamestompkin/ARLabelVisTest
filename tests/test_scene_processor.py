"""Smoke test for scene.process_video: build a synthetic scene + mask + LUT,
run the pipeline, and verify the CSV is well-formed and metrics.py can consume it.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import imageio.v3 as iio

from arlabelvis.viz import load_lut
from arlabelvis.scene import process_scene_video, tertile_bins, characteristic_env_color

HERE = Path(__file__).parent
OUT = HERE / "_smoke_out"
OUT.mkdir(exist_ok=True)


def make_synth_video(path: Path, n_frames: int = 30, w: int = 256, h: int = 256):
    """A scene whose background slowly shifts across hue — introduces real CEC motion."""
    frames = []
    for i in range(n_frames):
        # Hue cycles over time
        hue = i / n_frames  # 0 .. 1
        # Simple gradient: background fills the whole frame in hsv(hue, 0.8, 0.8), then a label patch in white
        from matplotlib.colors import hsv_to_rgb
        bg = (hsv_to_rgb(np.stack([np.full((h, w), hue),
                                   np.full((h, w), 0.8),
                                   np.full((h, w), 0.8)], axis=-1)) * 255).astype(np.uint8)
        frame = bg.copy()
        # Draw a white label patch in the center
        cy, cx = h // 2, w // 2
        frame[cy-30:cy+30, cx-60:cx+60] = (255, 255, 255)
        frames.append(frame)

    iio.imwrite(path, np.stack(frames, axis=0), fps=30, plugin="FFMPEG", codec="libx264")


def make_label_mask(path: Path, w: int = 256, h: int = 256):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//2 - 30 : h//2 + 30, w//2 - 60 : w//2 + 60] = 255
    iio.imwrite(path, mask)


def synth_lut_files() -> tuple[Path, Path]:
    """Synthetic complement LUT, same as viz smoke test. Reuses text format."""
    from skimage.color import rgb2lab
    step = 16
    grid = np.arange(0, 256, 256 // step, dtype=np.int32)
    R, G, B = np.meshgrid(grid, grid, grid, indexing="ij")
    rgb_in = np.stack([R, G, B], axis=-1).reshape(-1, 3)
    rgb_out = 255 - rgb_in
    lab_out = rgb2lab(rgb_out.astype(np.float32) / 255.0)
    lab_file = OUT / "synth_lab.txt"
    rgb_file = OUT / "synth_rgb.txt"
    np.savetxt(lab_file, lab_out, delimiter=",", fmt="%.6f")
    np.savetxt(rgb_file, rgb_in, delimiter=",", fmt="%d")
    return lab_file, rgb_file


def unit_checks():
    # tertile_bins sanity: (0,0,0) -> bin 0; (255,255,255) -> bin 26; (100,100,100) -> bin 13 (1,1,1)
    px = np.array([[0, 0, 0], [255, 255, 255], [100, 100, 100], [85, 170, 255]], dtype=np.uint8)
    b = tertile_bins(px)
    assert b[0] == 0, b[0]
    assert b[1] == 26, b[1]
    assert b[2] == 1*9 + 1*3 + 1, b[2]     # (1,1,1) tertile = 13
    assert b[3] == 0*9 + 1*3 + 2, b[3]     # (0,1,2) tertile = 5
    print("[PASS] tertile_bins")

    # CEC on a uniform red patch returns ~red
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

    lab_file, rgb_file = synth_lut_files()
    lut = load_lut(lab_file, rgb_file, value="lab")

    csv_out = OUT / "label_colors_export.csv"
    print(f"running pipeline...")
    n = process_scene_video(video_path, lut, mask_path, csv_out,
                            cec_source="background")
    assert n == 30, f"expected 30 rows, got {n}"

    # Verify metrics.py can read it
    print("checking metrics.py can consume the CSV...")
    import sys
    sys.path.insert(0, str(HERE.parent))  # so 'utils' package resolves
    from arlabelvis.metrics import load_and_filter, compute_gradients, print_stats
    df = load_and_filter(csv_out)
    grads = compute_gradients(df)
    print_stats(df, grads)
    print("[PASS] metrics.py roundtrip")
    print(f"\ndone. outputs in {OUT}")


if __name__ == "__main__":
    main()
