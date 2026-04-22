"""CLI driver for arlabelvis.scene.process_scene_video.
Replaces the Unity RenderStereoBackgroundforAreaLabel pipeline."""
import argparse

from arlabelvis.scene import process_scene_video
from arlabelvis.viz import load_lut


def main():
    p = argparse.ArgumentParser(description="Per-frame CEC + LUT lookup over a video.")
    p.add_argument("--video", required=True, help="video file or dir of frames")
    p.add_argument("--label-mask", required=True, help="PNG mask of label region")
    p.add_argument("--background-mask", default=None, help="optional bg mask (else = NOT label)")
    p.add_argument("--lab-file", required=True, help="AllCandidateLABvals_*.txt")
    p.add_argument("--rgb-file", required=True, help="AllCorrespondingRGBVals_*.txt")
    p.add_argument("--lut-value", default="lab", choices=["lab", "oklab", "rgb"])
    p.add_argument("--cec-source", default="label", choices=["label", "background"])
    p.add_argument("--num-top-bins", type=int, default=2)
    p.add_argument("-o", "--output", required=True, help="output CSV")
    args = p.parse_args()

    lut = load_lut(args.lab_file, args.rgb_file, value=args.lut_value)
    process_scene_video(
        args.video, lut, args.label_mask, args.output,
        background_mask_path=args.background_mask,
        cec_source=args.cec_source,
        num_top_bins=args.num_top_bins,
    )


if __name__ == "__main__":
    main()
