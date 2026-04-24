"""Extract high-quality 15.22s clips from the Dubai Marina timelapse at
the exact time windows the compressed-YouTube Dubai_* augmented videos came
from (per data/scenes/_clip_locations.json).

Produces two new scenes (full 3840x2160 frames kept; no crop):
  - data/scenes/dubai_dusk/       (frame 4064..4519 of the original)
  - data/scenes/dubai_daytime/    (frame 861..1316 of the original)

Both scenes share the label mask under dubai_changing_bgcolors/label_mask.png
via relative reference; arlabelvis.scene.process_scene_video center-pads the
mask to the 2160-tall frame automatically.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = REPO_ROOT / "data/scenes/dubai_marina_timelapse/video.mp4"
LOCATIONS = REPO_ROOT / "data/scenes/_clip_locations.json"
N_FRAMES = 456            # matches the two YouTube-compressed clips
FPS = 29.97
CRF = 15                  # near-lossless h.264

DEST = {
    # new_scene_name    -> corresponding augmented scene name in _clip_locations
    "dubai_dusk":    "dubai_changing_bgcolors",
    "dubai_daytime": "dubai_raining",
}


def _extract(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Accurate seek: -ss before -i for fast coarse seek, -ss after -i to refine,
    # then re-encode at high quality. `-an` drops audio (none in source anyway).
    coarse = max(0.0, start_s - 2.0)
    fine = start_s - coarse
    cmd = [ffmpeg, "-y",
           "-ss", f"{coarse:.6f}", "-i", str(src), "-ss", f"{fine:.6f}",
           "-t", f"{duration_s:.6f}",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
           "-pix_fmt", "yuv420p", "-an",
           str(dst)]
    print(f"[extract] {src.name} @ {start_s:.3f}s+{duration_s:.3f}s -> {dst.relative_to(REPO_ROOT)}")
    subprocess.run(cmd, check=True, capture_output=True)


def _write_manifest(scene_name: str, src_scene: str, start_frame: int,
                    start_time_s: float, duration_s: float) -> None:
    scene_dir = REPO_ROOT / "data/scenes" / scene_name
    mf = scene_dir / "manifest.json"
    meta = {
        "name": scene_name,
        "title": {
            "dubai_dusk":    "Dubai skyline, dusk -> night",
            "dubai_daytime": "Dubai skyline, daytime",
        }[scene_name],
        "description": {
            "dubai_dusk":    ("15.22s dusk-to-night slice of the Dubai Marina "
                              "timelapse (original YouTube source, not the "
                              "compressed augmented clip). CEC changes "
                              "naturally as the sky transitions — useful as "
                              "a non-adversarial stress test for LUT "
                              "stability."),
            "dubai_daytime": ("15.22s daytime slice of the Dubai Marina "
                              "timelapse. Stable CEC; complements dubai_dusk "
                              "as the 'easy' regime."),
        }[scene_name],
        "video": "video.mp4",
        "label_mask": "../dubai_changing_bgcolors/label_mask.png",
        "resolution": [3840, 2160],
        "projection": "equirectangular_360",
        "fps": FPS,
        "duration_s": round(duration_s, 3),
        "label_region": {
            "shape": "rect",
            "note": ("Shares dubai_changing_bgcolors/label_mask.png (3840x1920). "
                     "arlabelvis.scene.process_scene_video center-pads the mask "
                     "with 120 zero-rows top+bottom to match this 3840x2160 "
                     "source; the resulting label bbox is at y in [1033, 1307] "
                     "of the frame.")
        },
        "source": "data/scenes/dubai_marina_timelapse/video.mp4",
        "source_slice": {
            "start_frame_in_original": start_frame,
            "start_time_s_in_original": round(start_time_s, 3),
            "duration_s": round(duration_s, 3),
            "derived_from_augmented": src_scene,
        },
        "notes": [
            "Extracted via ffmpeg CRF=15 libx264 from the original; no "
            "re-compression artefacts beyond that single encoding pass.",
        ],
    }
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    print(f"[manifest] {mf.relative_to(REPO_ROOT)}")


def main():
    locations = json.loads(LOCATIONS.read_text(encoding="utf-8"))
    duration_s = N_FRAMES / FPS
    for new_name, src_name in DEST.items():
        r = locations["results"][src_name]
        start_frame = r["start_frame"]
        start_time_s = start_frame / FPS
        scene_dir = REPO_ROOT / "data/scenes" / new_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        _extract(ORIGINAL, scene_dir / "video.mp4", start_time_s, duration_s)
        _write_manifest(new_name, src_name, start_frame, start_time_s, duration_s)


if __name__ == "__main__":
    main()
