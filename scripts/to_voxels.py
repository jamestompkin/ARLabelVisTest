"""DEPRECATED — fed the external binvox-trained neural bounding pipeline.

The paper pipeline now uses the in-process MLP in
``arlabelvis.neural_bounding`` (see ``shape='neural'`` in
``arlabelvis.luts.LutConfig``); no binvox round-trip is involved. This
script remains only for archival reference.
"""
import argparse
from pathlib import Path

from arlabelvis.colors import srgb_to_oklab
from arlabelvis.meshing import generate_labs
from arlabelvis.voxels import write_voxels


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space", default="CIELAB", choices=["CIELAB", "OKLAB", "sRGB"])
    p.add_argument("--interval", type=int, default=1,
                   help="interval for the 256^3 sRGB input grid (default: 1)")
    p.add_argument("--dim", type=int, default=256, help="voxel resolution")
    args = p.parse_args()

    all_rgbs, all_cielabs = generate_labs(interval=args.interval)
    if args.space == "CIELAB":
        points = all_cielabs
    elif args.space == "OKLAB":
        points = srgb_to_oklab(all_rgbs)
    else:
        points = all_rgbs

    out = Path(f"external/neural_bounding/data/3D/"
               f"{args.space}_{args.interval}_{args.dim}.binvox")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_voxels(points, args.dim, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
