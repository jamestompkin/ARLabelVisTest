"""Registry of every figure/table in the paper and the LUT configs it needs.

Kept in one place so:
  - `reproduce_all.py` can iterate every figure script and check for outputs;
  - LUT cache entries get hit by multiple figure scripts (e.g., the α̂=0.25
    CIELAB RGD LUT is referenced by `fig_distance_comparison`,
    `fig_alpha_sweep_cubes`, and `fig_neural_bounded_comparison`).

Figure/table IDs match the LaTeX `\\label{...}` in the paper sources where
possible (e.g., `fig_srgb_euclidean` ↔ `\\label{fig:srgb_euclidean}`).
"""
from __future__ import annotations

from scripts.paper._lut_cache import LutConfig


# Shorthand constructors
def CIELAB(**kw): return LutConfig(space="CIELAB", **kw)
def OKLAB(**kw): return LutConfig(space="OKLAB", **kw)
def SRGB(**kw): return LutConfig(space="sRGB", **kw)


# --- Core named LUTs that several figures share ------------------------------
# Euclidean in sRGB reduces to a degenerate 8-cube-corner map: each input
# picks the opposite corner of the 0-255 sRGB cube. Included in tables as the
# baseline demonstration of why picking "farthest color" without a perceptual
# transform doesn't work.
LUT_SRGB_EUCLIDEAN      = SRGB(metric="Euclidean", smoothing="none")
LUT_CIELAB_EUCLIDEAN    = CIELAB(metric="Euclidean", smoothing="none")
# Euclidean-in-CIELAB with CIELAB-space output interpolation (Kwon 2019's
# baseline). ΔE₇₆ is by definition Euclidean distance in CIELAB, so this is
# computed via the Euclidean path — the distinguishing choice is where the
# output LUT interpolates.
LUT_CIELAB_DELTAE76     = CIELAB(metric="Euclidean", smoothing="none",
                                 interp_space="CIELAB")
LUT_CIELAB_DELTAE94     = CIELAB(metric="DeltaE94", smoothing="none",
                                 interp_space="CIELAB")
LUT_CIELAB_DELTAE00     = CIELAB(metric="DeltaE00", smoothing="none",
                                 interp_space="CIELAB")
LUT_OKLAB_EUCLIDEAN     = OKLAB(metric="Euclidean", smoothing="none")

# Bounded-gamut variants (mesh-based smoothing)
LUT_CIELAB_SPHERE_RGD_025 = CIELAB(smoothing="sphere",      metric="RGD", alpha_hat=0.25)
LUT_CIELAB_NEURAL_RGD_025 = CIELAB(smoothing="neural",      metric="RGD", alpha_hat=0.25)
LUT_CIELAB_HULL_RGD_025   = CIELAB(smoothing="convex_hull", metric="RGD", alpha_hat=0.25)

# α̂=0.05 variants used by the short paper's space-comparison and teaser
# figures. The neural α̂=0.05 LUT is also the first entry of LUT_ALPHA_SWEEP.
LUT_CIELAB_HULL_RGD_005   = CIELAB(smoothing="convex_hull", metric="RGD", alpha_hat=0.05)
LUT_OKLAB_HULL_RGD_005    = OKLAB(smoothing="convex_hull",  metric="RGD", alpha_hat=0.05)
LUT_CIELAB_NEURAL_RGD_005 = CIELAB(smoothing="neural",      metric="RGD", alpha_hat=0.05)
# α̂=0.5 neural variant used by the short paper's "ours" hue histogram
# (filename suffix 50 in thesis legacy naming).
LUT_CIELAB_NEURAL_RGD_050 = CIELAB(smoothing="neural",      metric="RGD", alpha_hat=0.5)
# α̂=1.25 is the short paper's hero method (see §Method and tab:scene).
LUT_CIELAB_NEURAL_RGD_125 = CIELAB(smoothing="neural",      metric="RGD", alpha_hat=1.25)

# alpha_hat sweep on the neural-bounded mesh (thesis uses 0.05..1.5)
LUT_ALPHA_SWEEP = [
    CIELAB(smoothing="neural", metric="RGD", alpha_hat=a)
    for a in (0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5)
]

# Gaussian-smoothed variants at thesis-mentioned sigmas
LUT_CIELAB_GAUSSIAN_SIGMAS = [
    CIELAB(smoothing="gaussian", sigma=s, metric="RGD", alpha_hat=0.05)
    for s in (0.25, 2.0, 4.0)
]


# --- Figure/table registry ---------------------------------------------------
# Each entry: "figure_id" -> dict(module=..., outputs=[...], luts=[...])
FIGURES = {
    # === thesis chapter 4 (methodology/results) ===
    "fig_srgb_euclidean": dict(
        module="scripts.paper.fig_srgb_euclidean",
        outputs=["results/paper/figures/srgb_euclidean.png"],
        luts=[LUT_SRGB_EUCLIDEAN],
    ),
    "fig_cielab_euclidean": dict(
        module="scripts.paper.fig_cielab_euclidean",
        outputs=["results/paper/figures/cielab_euclidean.png"],
        luts=[LUT_CIELAB_EUCLIDEAN],
    ),
    "fig_color_space_comparison": dict(
        module="scripts.paper.fig_color_space_comparison",
        outputs=["results/paper/figures/color_space_comparison.png"],
        luts=[LUT_OKLAB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN],
    ),
    "fig_sphere": dict(
        module="scripts.paper.fig_sphere_lookup",
        outputs=["results/paper/figures/sphere_lookup.png"],
        luts=[LUT_CIELAB_SPHERE_RGD_025],
    ),
    "fig_neural_bounded_lut": dict(
        module="scripts.paper.fig_neural_bounded_lut",
        outputs=["results/paper/figures/neural_bounded_lut.png"],
        luts=[LUT_CIELAB_NEURAL_RGD_025],
    ),
    "fig_gaussian_smoothing": dict(
        module="scripts.paper.fig_gaussian_smoothing",
        outputs=["results/paper/figures/gaussian_smoothing.png"],
        luts=LUT_CIELAB_GAUSSIAN_SIGMAS,
    ),
    "fig_distance_comparison": dict(
        module="scripts.paper.fig_distance_comparison",
        outputs=["results/paper/figures/distance_comparison.png"],
        luts=[LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_DELTAE76, LUT_CIELAB_HULL_RGD_025],
    ),
    "fig_alpha_sweep_cubes": dict(
        module="scripts.paper.fig_alpha_sweep_cubes",
        outputs=["results/paper/figures/alpha_sweep_cubes.png"],
        luts=LUT_ALPHA_SWEEP,
    ),
    "fig_alpha_plots": dict(
        module="scripts.paper.fig_alpha_plots",
        outputs=["results/paper/figures/alpha_plots.png"],
        luts=LUT_ALPHA_SWEEP,
    ),
    "fig_hue_histograms": dict(
        module="scripts.paper.fig_hue_histograms",
        outputs=["results/paper/figures/hue_histogram_euclidean.png",
                 "results/paper/figures/hue_histogram_deltae76.png",
                 "results/paper/figures/hue_histogram_rgd.png"],
        luts=[LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_DELTAE76, LUT_CIELAB_NEURAL_RGD_025],
    ),
}

TABLES = {
    "tab_space_and_interp": dict(
        module="scripts.paper.tab_space_and_interp",
        outputs=["results/paper/tables/space_and_interp.tex"],
        luts=[LUT_SRGB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN,
              LUT_CIELAB_DELTAE76, LUT_OKLAB_EUCLIDEAN],
    ),
    "tab_geometry_smoothing": dict(
        module="scripts.paper.tab_geometry_smoothing",
        outputs=["results/paper/tables/geometry_smoothing.tex"],
        luts=[CIELAB(smoothing="convex_hull", metric="RGD", alpha_hat=0.05),
              *LUT_CIELAB_GAUSSIAN_SIGMAS,
              LUT_CIELAB_NEURAL_RGD_025],
    ),
    "tab_distance_metric": dict(
        module="scripts.paper.tab_distance_metric",
        outputs=["results/paper/tables/distance_metric.tex"],
        luts=[LUT_CIELAB_DELTAE76, LUT_CIELAB_DELTAE94, LUT_CIELAB_DELTAE00,
              CIELAB(smoothing="convex_hull", metric="RGD",
                     alpha_hat=0.25, interp_space="CIELAB")],
    ),
}


# --- IEEE VIS 2026 short paper ----------------------------------------------
# Outputs land at scripts.paper._shared.SHORTPAPER_FIG_DIR (the Overleaf
# ieeevis2026/figures directory), under the exact filenames that paper.tex
# references via \includegraphics. Filenames match the thesis's historical
# naming where the short paper reuses thesis figure slots.
_SP = "scripts.paper"

def _sp_out(*names):
    # reproduce_all uses these to skip scripts whose outputs already exist.
    # Short-paper outputs live outside the repo, so path checks go through
    # SHORTPAPER_FIG_DIR at query time (see reproduce_all).
    return [f"shortpaper:{n}" for n in names]

SHORTPAPER_FIGURES = {
    "shortpaper_teaser": dict(
        module=f"{_SP}.shortpaper_teaser",
        outputs=_sp_out("teaser/old_lookup.png",
                        "teaser/neural_bounding_05.png"),
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_005],
    ),
    "shortpaper_space_comparison": dict(
        module=f"{_SP}.shortpaper_space_comparison",
        outputs=_sp_out("space_comparison/rgb_euclidean.png",
                        "space_comparison/oklab_rgd_05.png",
                        "space_comparison/cielab_rgd_05.png"),
        luts=[LUT_SRGB_EUCLIDEAN, LUT_OKLAB_HULL_RGD_005, LUT_CIELAB_HULL_RGD_005],
    ),
    "shortpaper_smoothing_zooms": dict(
        module=f"{_SP}.shortpaper_smoothing_zooms",
        outputs=_sp_out("smoothing/zoom_4o0.png",
                        "smoothing/zoom_neural.png"),
        luts=[LUT_CIELAB_NEURAL_RGD_005, *LUT_CIELAB_GAUSSIAN_SIGMAS],
    ),
    "shortpaper_hue_histograms": dict(
        module=f"{_SP}.shortpaper_hue_histograms",
        outputs=_sp_out("hue_histograms/OriginalLABVals_hue.png",
                        "hue_histograms/AllCandidateLABvals_CIELAB_1_Euclidean_hue.png",
                        "hue_histograms/AllCandidateLABvals_CIELAB_1_RGD_50_neural_256_hue.png"),
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_NEURAL_RGD_050],
    ),
    "shortpaper_alpha_plots": dict(
        module=f"{_SP}.shortpaper_alpha_plots",
        outputs=_sp_out("alpha_plots/alpha_plots.png"),
        luts=LUT_ALPHA_SWEEP,
    ),
}


def _sp_tab_out(*names):
    return [f"shortpaper_tab:{n}" for n in names]


SHORTPAPER_TABLES = {
    "shortpaper_tab_scene": dict(
        module=f"{_SP}.shortpaper_tab_scene",
        outputs=_sp_tab_out("scene/scene.tex"),
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_125],
    ),
    "shortpaper_tab_timing": dict(
        module=f"{_SP}.shortpaper_tab_timing",
        outputs=_sp_tab_out("timing/timing.tex"),
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_125],
    ),
}
