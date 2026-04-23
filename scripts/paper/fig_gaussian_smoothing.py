"""Figure: LUTs with Gaussian smoothing at σ ∈ {0.25, 2.0, 4.0} (thesis fig:gaussian_smoothing)."""
from PIL import Image
from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import LUT_CIELAB_GAUSSIAN_SIGMAS
from scripts.paper._lut_cache import get_lut
from scripts.paper._shared import FIG_DIR, lut_to_srgb_u8

OUT = FIG_DIR / "gaussian_smoothing.png"


def main():
    tmp = FIG_DIR / "_tmp_gauss"
    tmp.mkdir(exist_ok=True)
    paths = []
    for cfg in LUT_CIELAB_GAUSSIAN_SIGMAS:
        tag = f"sigma_{cfg.sigma:g}".replace(".", "p")
        p = tmp / f"{tag}.png"
        render_srgb_cube_isometric(lut_to_srgb_u8(get_lut(cfg), cfg), save_path=p,
                                  title=f"σ={cfg.sigma:g}")
        paths.append(p)

    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(im.height for im in imgs)
    w_total = sum(im.width for im in imgs)
    out = Image.new("RGB", (w_total, h), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, 0)); x += im.width
    out.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
