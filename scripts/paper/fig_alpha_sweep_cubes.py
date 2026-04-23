"""Figure: cubes across the α̂ sweep (thesis fig:rgd_vals).

Arranges the 9 LUTs in a 3x3 grid.
"""
from PIL import Image
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_ALPHA_SWEEP
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "alpha_sweep_cubes.png"


def main():
    tmp = FIG_DIR / "_tmp_alpha"
    tmp.mkdir(exist_ok=True)

    paths = []
    for cfg in LUT_ALPHA_SWEEP:
        tag = f"a{cfg.alpha_hat:g}".replace(".", "p")
        p = tmp / f"{tag}.png"
        render_srgb_cube_isometric(lut_to_srgb_u8(get_lut(cfg), cfg), save_path=p,
                                  title=f"α̂ = {cfg.alpha_hat:g}")
        paths.append(p)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    cw = max(im.width for im in imgs)
    ch = max(im.height for im in imgs)
    out = Image.new("RGB", (cols * cw, rows * ch), "white")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out.paste(im, (c * cw, r * ch))
    out.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
