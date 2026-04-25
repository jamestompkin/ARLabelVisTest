"""End-to-end sanity check of the LUT pipeline. Runs in ~1 min from cold start.

Builds a small CIELAB/Euclidean farthest-color LUT via the paper pipeline's
``get_lut`` (no MATLAB, no neural bounding), renders the sRGB cube view of
it. Exercises every stage: ``generate_labs`` → ``build_candidates`` →
``score_argmax`` → ``render_candidate`` → ``interpolate_interval`` →
``lut_to_srgb_u8`` → ``render_srgb_cube_isometric``.

Output PNG lands at ``results/e2e_demo/lut_cube_cielab_euclidean.png``.
"""
import logging
from pathlib import Path

from arlabelvis.luts import LutConfig, get_lut, lut_to_srgb_u8
from arlabelvis.viz import render_srgb_cube_isometric

OUT = Path("results/e2e_demo/lut_cube_cielab_euclidean.png")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = LutConfig(working_space="CIELAB", smoothing="none", metric="Euclidean",
                    interval=8, output_space="sRGB")
    lut_u8 = lut_to_srgb_u8(get_lut(cfg), cfg.output_space)
    render_srgb_cube_isometric(
        lut_u8, save_path=OUT,
        title="CIELAB Euclidean farthest-color LUT (interval=8)",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
