"""Render a hue-distribution histogram of a lookup table."""
import argparse

from arlabelvis.viz import load_lut, render_hue_histogram


def main():
    p = argparse.ArgumentParser(description="Render LUT hue histogram.")
    p.add_argument("lab_file", help="AllCandidateLABvals_*.txt")
    p.add_argument("rgb_file", help="AllCorrespondingRGBVals_*.txt")
    p.add_argument("-o", "--out", required=True, help="output PNG")
    p.add_argument("--value", default="lab", choices=["lab", "oklab", "rgb"])
    p.add_argument("--bins", type=int, default=360)
    args = p.parse_args()

    lut = load_lut(args.lab_file, args.rgb_file, value=args.value)
    render_hue_histogram(lut, save_path=args.out, bins=args.bins)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
