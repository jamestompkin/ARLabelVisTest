"""Render a paper-ready isometric RGB-cube view of a lookup table."""
import argparse

from arlabelvis.viz import load_lut, render_rgb_cube_isometric


def main():
    p = argparse.ArgumentParser(description="Render LUT as an isometric RGB cube.")
    p.add_argument("lab_file", help="AllCandidateLABvals_*.txt (or oklab/rgb file)")
    p.add_argument("rgb_file", help="AllCorrespondingRGBVals_*.txt")
    p.add_argument("-o", "--out", required=True, help="output PNG")
    p.add_argument("--value", default="lab", choices=["lab", "oklab", "rgb"])
    p.add_argument("--stride", type=int, default=2, help="face-downsample stride")
    args = p.parse_args()

    lut = load_lut(args.lab_file, args.rgb_file, value=args.value)
    render_rgb_cube_isometric(lut, save_path=args.out, stride=args.stride)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
