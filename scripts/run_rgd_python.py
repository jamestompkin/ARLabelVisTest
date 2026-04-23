"""Python replacement for MATLAB `external/matlab_rgd/demo.m`.

Reads an OFF mesh, runs all-pairs regularized geodesic distances via the
Python ADMM port, writes the same `max_indices_*.txt` file the existing
pipeline consumes.

Drop-in with MATLAB: output file is a single comma-separated row of 1-indexed
argmaxes, byte-identical to what MATLAB's
`writematrix(max_indices, final_file)` produces.

Two modes:

  scripts.run_rgd_python --from-config
    Reads the RunConfig in this module's CONFIG block to derive .off input
    and max_indices output paths (matches what MATLAB demo.m + scripts.run_rgd
    expect so the downstream scripts.full_pipeline works unchanged).

  scripts.run_rgd_python <off_file> -o <output>  [--alpha-hat ...]
    Free-form mode; point it at any OFF and output wherever.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from arlabelvis.rgd.allpairs import compute_all_pairs_argmax, read_off
from scripts._config import RunConfig


# Single source of truth: the algorithm's actual float alpha_hat.
# RunConfig.alpha is a stringy filename tag — derived from ALPHA_HAT here so
# the two can't drift out of sync.
ALPHA_HAT = 1.25
CONFIG = RunConfig(interval=1, space="CIELAB", distance_measure="RGD",
                   alpha=f"{int(round(ALPHA_HAT * 100))}",
                   smoothing_mode="neural", voxel_dim=256)


def main():
    p = argparse.ArgumentParser(description="All-pairs RGD via Python (MATLAB demo.m replacement).")
    p.add_argument("off_file", nargs="?", default=None,
                   help="input OFF mesh. omit with --from-config to use the RunConfig paths.")
    p.add_argument("-o", "--output", default=None,
                   help="output max_indices file. omit with --from-config to use RunConfig.matlab_indices_path.")
    p.add_argument("--alpha-hat", type=float, default=None,
                   help="regularizer weight (default: CONFIG_ALPHA_HAT from this script).")
    p.add_argument("--from-config", action="store_true",
                   help="use this script's CONFIG for paths (drop-in demo.m replacement).")
    p.add_argument("--workers", type=int, default=None,
                   help="multiprocessing pool size; default os.cpu_count()-1.")
    args = p.parse_args()

    if args.from_config:
        off_file = CONFIG.original_off_path
        output = CONFIG.matlab_indices_path
        alpha_hat = args.alpha_hat if args.alpha_hat is not None else ALPHA_HAT
        print(f"[from-config] .off = {off_file}")
        print(f"[from-config] out = {output}")
    else:
        if args.off_file is None or args.output is None:
            p.error("off_file and --output required unless --from-config is set")
        off_file = args.off_file
        output = args.output
        alpha_hat = args.alpha_hat if args.alpha_hat is not None else ALPHA_HAT

    V, F = read_off(off_file)
    print(f"read {off_file}: {len(V)} verts, {len(F)} faces")

    max_indices = compute_all_pairs_argmax(V, F, alpha_hat,
                                            n_workers=args.workers,
                                            one_indexed=True)

    # Match MATLAB's writematrix format: single comma-separated row of ints.
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, max_indices.reshape(1, -1), fmt="%d", delimiter=",")
    print(f"wrote {out}  ({len(max_indices)} argmaxes)")


if __name__ == "__main__":
    main()
