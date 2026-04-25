"""Short-paper fig:smoothing: zoom-in comparison of Gaussian vs neural smoothing.

Both panels show the same sub-region of the LUT cube so local texture can be
compared at matching RGD regularisation (\\hat\\alpha = 0.05). The short paper
uses these to argue that the neural bound introduces more intermediate hue
transitions per unit boundary length than matching-level Gaussian smoothing.

Outputs into SHORTPAPER_FIG_DIR:
  - zoom_4o0.png       Gaussian sigma=4.0 cube, cropped center patch
  - zoom_neural.png    neural-bounded cube, cropped center patch

Crop region is centered on the front R-face at moderate G/B values — a region
where the LUT shows visible colour transitions under both smoothings. If a
different region is wanted, adjust CROP_BBOX.
"""
from PIL import Image

from arlabelvis.viz import render_srgb_cube_isometric

from scripts.paper._configs import (LUT_CIELAB_NEURAL_RGD_005,
                                    LUT_CIELAB_GAUSSIAN_SIGMAS)
from arlabelvis.luts import get_lut
from arlabelvis.luts import lut_to_srgb_u8
from scripts.paper._paths import SHORTPAPER_FIG_DIR, fig_path


# (left, top, right, bottom) as fractions of the full cube render.
# Picks a centered square covering ~50% of the frame.
CROP_BBOX_FRAC = (0.25, 0.25, 0.75, 0.75)


def _render_and_crop(cfg, out_name: str, title: str):
    tmp = SHORTPAPER_FIG_DIR / "_tmp_shortpaper_zoom"
    tmp.mkdir(exist_ok=True)
    full = tmp / f"{out_name}.full.png"
    render_srgb_cube_isometric(
        lut_to_srgb_u8(get_lut(cfg), cfg.output_space), save_path=full, title=title,
    )
    img = Image.open(full).convert("RGB")
    w, h = img.size
    left, top, right, bottom = CROP_BBOX_FRAC
    crop = img.crop((int(w * left), int(h * top),
                     int(w * right), int(h * bottom)))
    crop.save(fig_path("smoothing", out_name))
    print(f"wrote smoothing/{out_name}")


def main():
    sigma_4 = next(c for c in LUT_CIELAB_GAUSSIAN_SIGMAS
                   if c.shape_gaussian_sigma == 4.0)
    _render_and_crop(sigma_4, "zoom_4o0.png",
                     r"Gaussian $\sigma=4.0$ ($\hat\alpha=0.05$)")
    _render_and_crop(LUT_CIELAB_NEURAL_RGD_005, "zoom_neural.png",
                     r"Neural-bounded ($\hat\alpha=0.05$)")


if __name__ == "__main__":
    main()
