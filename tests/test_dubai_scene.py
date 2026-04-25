"""Reproduce the flicker-analysis pipeline end-to-end on the dubai_changing_bgcolors scene.

What this exercises:
  - `arlabelvis.scene_catalog.load_scene` resolves the manifest + files correctly.
  - `arlabelvis.scene_video.process_scene_video` decodes the real 3840×1920 equirectangular
    stereo video, masks the label region, computes CEC + LUT lookup per frame, and
    writes a CSV.
  - `arlabelvis.metrics.{load_and_filter, compute_gradients, print_stats}` can
    consume that CSV and produce the flicker-gradient stats.

LUT selection, in order of preference (first one that resolves is used):
  1. A paper-pipeline cached LUT chosen via `--lut` (neural_rgd / cielab_euclidean
     / rgb_euclidean / deltae76 / deltae94 / deltae00). Defaults to neural_rgd.
  2. Falls back to `cielab_euclidean` if the chosen LUT isn't cached yet.
  3. Finally a synthetic identity-complement LUT so the script always runs.

Flags:
  --max-frames N   cap at first N frames (useful for fast iteration; default: full).
  --lut NAME       pick which cached LUT to drive the lookup with.
  --out-dir PATH   override where the CSV lands (default: tests/_smoke_out/dubai/).

Run:
  uv run python -m tests.test_dubai_scene
  uv run python -m tests.test_dubai_scene --max-frames 60 --lut deltae76
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from arlabelvis.scene_catalog import load_scene
from arlabelvis.scene_video import process_scene_video
from arlabelvis.metrics import load_and_filter, compute_gradients, print_stats


HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "_smoke_out" / "dubai"


# Map friendly LUT nicknames to paper-pipeline config objects. Resolved lazily
# so importing this test doesn't require the paper package to exist.
def _paper_lut_configs():
    from scripts.paper._configs import (
        LUT_SRGB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN,
        LUT_CIELAB_DELTAE76, LUT_CIELAB_DELTAE94, LUT_CIELAB_DELTAE00,
        LUT_CIELAB_NEURAL_RGD_025,
    )
    return {
        "rgb_euclidean":   LUT_SRGB_EUCLIDEAN,
        "cielab_euclidean": LUT_CIELAB_EUCLIDEAN,
        "deltae76":        LUT_CIELAB_DELTAE76,
        "deltae94":        LUT_CIELAB_DELTAE94,
        "deltae00":        LUT_CIELAB_DELTAE00,
        "neural_rgd":      LUT_CIELAB_NEURAL_RGD_025,
    }


def _synth_lut():
    """256³ complement-in-RGB LUT. Always available; used if no cached LUT
    is found so the test still exercises the full pipeline."""
    print("[lut] falling back to synthetic complement-in-RGB LUT")
    lut = np.empty((256, 256, 256, 3), dtype=np.uint8)
    r = np.arange(256, dtype=np.uint8)
    lut[..., 0] = 255 - r[:, None, None]
    lut[..., 1] = 255 - r[None, :, None]
    lut[..., 2] = 255 - r[None, None, :]
    return lut, "synthetic-complement"


def _resolve_lut(nick: str):
    """Return (lut_u8, label). Prefers cached paper LUTs; falls back in order."""
    try:
        from arlabelvis.luts import get_lut
        from arlabelvis.luts import lut_to_srgb_u8
    except ImportError as e:
        print(f"[lut] paper cache unavailable ({e}); using synthetic")
        return _synth_lut()

    cfgs = _paper_lut_configs()
    preferred = [nick, "cielab_euclidean", "rgb_euclidean"]
    tried: list[str] = []
    for name in preferred:
        if name in tried or name not in cfgs:
            continue
        tried.append(name)
        cfg = cfgs[name]
        from arlabelvis.luts import DEFAULT_CACHE
        if not DEFAULT_CACHE.has(cfg):
            print(f"[lut] {name}: not cached at {DEFAULT_CACHE.path_for(cfg).name}")
            continue
        return lut_to_srgb_u8(get_lut(cfg), cfg.output_space), name

    return _synth_lut()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="dubai_changing_bgcolors")
    p.add_argument("--lut", default="neural_rgd",
                   choices=["rgb_euclidean", "cielab_euclidean",
                            "deltae76", "deltae94", "deltae00", "neural_rgd"])
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap at first N frames (default: full video)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    print(f"=== loading scene {args.scene!r} ===")
    scene = load_scene(args.scene)
    print(f"  video:      {scene.video.relative_to(scene.video.parents[3])}")
    print(f"  label mask: {scene.label_mask.relative_to(scene.label_mask.parents[3])}")
    print(f"  {scene.resolution[0]}x{scene.resolution[1]} @ {scene.fps:.2f} fps, "
          f"{scene.duration_s:.2f} s")

    assert scene.label_mask is not None, f"{args.scene} has no label mask in its manifest"

    lut, lut_label = _resolve_lut(args.lut)
    print(f"=== driving with LUT: {lut_label} ===")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / f"{args.scene}__{lut_label}.csv"

    print(f"=== running processor -> {csv_out.name} ===")
    if args.max_frames is not None:
        _run_with_frame_cap(scene, lut, csv_out, args.max_frames)
    else:
        process_scene_video(str(scene.video), lut, str(scene.label_mask), str(csv_out),
                            cec_source="label")

    print(f"=== loading CSV + computing flicker gradients ===")
    df = load_and_filter(csv_out)
    n_rows = len(df)
    assert n_rows > 0, "CSV is empty"
    expected_full = int(scene.fps * scene.duration_s)
    if args.max_frames is None:
        assert abs(n_rows - expected_full) <= 5, \
            f"expected ~{expected_full} rows, got {n_rows}"
    else:
        assert n_rows == args.max_frames, \
            f"expected exactly {args.max_frames} rows, got {n_rows}"

    grads = compute_gradients(df)
    print_stats(df, grads)

    print(f"\n[PASS] dubai scene roundtrip ({n_rows} frames, LUT={lut_label})")
    print(f"       CSV: {csv_out}")
    return 0


def _run_with_frame_cap(scene, lut, csv_out, max_frames: int):
    """Monkey-patch iter_frames so process_scene_video early-terminates at
    max_frames. Keeps the main pipeline untouched for production use."""
    from arlabelvis import scene_video as scene_mod

    orig = scene_mod.iter_frames

    def capped(video_path):
        it, fps = orig(video_path)

        def gen():
            for i, frame in it:
                if i >= max_frames:
                    break
                yield i, frame
        return gen(), fps

    scene_mod.iter_frames = capped
    try:
        process_scene_video(str(scene.video), lut, str(scene.label_mask), str(csv_out),
                            cec_source="label")
    finally:
        scene_mod.iter_frames = orig


if __name__ == "__main__":
    sys.exit(main())
