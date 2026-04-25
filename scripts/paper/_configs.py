"""Registry of every figure/table in the short paper and the LUT configs it needs.

Kept in one place so:
  - ``reproduce_all.py`` can iterate every figure/table script and skip ones
    whose outputs already exist.
  - LUT cache entries get hit by multiple scripts (e.g., the neural+RGD
    α̂=0.05 LUT is referenced by the teaser, the space-comparison figure,
    and the smoothing-zoom figure).

Figure/table IDs match the LaTeX ``\\label{...}`` in ``ieeevis2026/paper.tex``
where possible (e.g., ``fig_teaser`` ↔ ``\\label{fig:teaser}``). Entry
``outputs`` are paths relative to ``SHORTPAPER_DIR`` — ``.png`` items land
in ``figures/``, ``.tex`` items in ``tables/`` (resolved by ``reproduce_all``).
"""
from __future__ import annotations

from arlabelvis.luts import LutConfig


# --- Named LUTs used across short-paper scripts ------------------------------
# Euclidean in sRGB reduces to a degenerate 8-cube-corner map. The "naive
# baseline" panel in fig_space_comparison.
LUT_SRGB_EUCLIDEAN = LutConfig(
    working_space="sRGB", shape="hull", metric="Euclidean",
)
# Kwon 2019's baseline: ΔE₀₀ argmax on the CIELAB convex hull, output
# interpolated in CIELAB.
LUT_CIELAB_DELTAE00 = LutConfig(
    working_space="CIELAB", shape="hull", metric="DeltaE00", output_space="CIELAB",
)
# CIELAB + Euclidean, sRGB-interp — the "CIELAB Euclidean" panel of
# fig_hue_histograms (mid-point between sRGB and ΔE-based baselines).
LUT_CIELAB_EUCLIDEAN = LutConfig(
    working_space="CIELAB", shape="hull", metric="Euclidean",
)

# Hull + RGD variants (convex-hull candidate set + regularised geodesic)
# used in fig_space_comparison.
LUT_CIELAB_HULL_RGD_005 = LutConfig(
    working_space="CIELAB", shape="hull", metric="RGD", metric_rgd_alpha_hat=0.05,
)
LUT_OKLAB_HULL_RGD_005 = LutConfig(
    working_space="OKLAB", shape="hull", metric="RGD", metric_rgd_alpha_hat=0.05,
)

# Sphere-bounded candidate set (outer bound via circumscribing icosphere).
LUT_CIELAB_SPHERE_RGD_005 = LutConfig(
    working_space="CIELAB", shape="sphere", metric="RGD", metric_rgd_alpha_hat=0.05,
)

# Gaussian-smoothed hull at a single sigma (for side-by-side palette comparison).
LUT_CIELAB_HULL_GAUSSIAN_RGD_005 = LutConfig(
    working_space="CIELAB", shape="hull", smoothing="gaussian",
    shape_gaussian_sigma=2.0, metric="RGD", metric_rgd_alpha_hat=0.05,
)

# Neural-bounded RGD: paper's "ours" method at three α̂ values.
LUT_CIELAB_NEURAL_RGD_005 = LutConfig(
    working_space="CIELAB", shape="neural", metric="RGD", metric_rgd_alpha_hat=0.05,
)
LUT_CIELAB_NEURAL_RGD_050 = LutConfig(
    working_space="CIELAB", shape="neural", metric="RGD", metric_rgd_alpha_hat=0.5,
)
LUT_CIELAB_NEURAL_RGD_125 = LutConfig(
    working_space="CIELAB", shape="neural", metric="RGD", metric_rgd_alpha_hat=1.25,
)

# α̂ sweep on the neural-bounded mesh for fig_alpha_plots.
LUT_ALPHA_SWEEP = [
    LutConfig(working_space="CIELAB", shape="neural", metric="RGD", metric_rgd_alpha_hat=a)
    for a in (0.05, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5)
]

# Gaussian-smoothed hull variants used by fig_smoothing_zooms.
LUT_CIELAB_GAUSSIAN_SIGMAS = [
    LutConfig(working_space="CIELAB", shape="hull", smoothing="gaussian", shape_gaussian_sigma=s,
              metric="RGD", metric_rgd_alpha_hat=0.05)
    for s in (0.25, 2.0, 4.0)
]


# --- Figure/table registry ---------------------------------------------------
# Each entry: "id" -> dict(module, outputs, luts). ``outputs`` are paths
# relative to SHORTPAPER_DIR; reproduce_all routes .png -> figures/,
# .tex -> tables/ by extension.

FIGURES = {
    "fig_teaser": dict(
        module="scripts.paper.fig_teaser",
        outputs=["teaser/old_lookup.png",
                 "teaser/neural_bounding_05.png"],
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_005],
    ),
    "fig_space_comparison": dict(
        module="scripts.paper.fig_space_comparison",
        outputs=["space_comparison/rgb_euclidean.png",
                 "space_comparison/oklab_rgd_05.png",
                 "space_comparison/cielab_rgd_05.png"],
        luts=[LUT_SRGB_EUCLIDEAN, LUT_OKLAB_HULL_RGD_005, LUT_CIELAB_HULL_RGD_005],
    ),
    "fig_smoothing_zooms": dict(
        module="scripts.paper.fig_smoothing_zooms",
        outputs=["smoothing/zoom_4o0.png",
                 "smoothing/zoom_neural.png"],
        luts=[LUT_CIELAB_NEURAL_RGD_005, *LUT_CIELAB_GAUSSIAN_SIGMAS],
    ),
    "fig_hue_histograms": dict(
        module="scripts.paper.fig_hue_histograms",
        outputs=["hue_histograms/OriginalLABVals_hue.png",
                 "hue_histograms/AllCandidateLABvals_CIELAB_1_Euclidean_hue.png",
                 "hue_histograms/AllCandidateLABvals_CIELAB_1_RGD_50_neural_256_hue.png"],
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_NEURAL_RGD_050],
    ),
    "fig_alpha_plots": dict(
        module="scripts.paper.fig_alpha_plots",
        outputs=["alpha_plots/alpha_plots.png"],
        luts=LUT_ALPHA_SWEEP,
    ),
    "fig_palette_geometry": dict(
        module="scripts.paper.fig_palette_geometry",
        outputs=["palette_geometry/palette_geometry.png",
                 "palette_geometry/srgb_euclidean.png",
                 "palette_geometry/cielab_euclidean.png",
                 "palette_geometry/kwon_de00.png",
                 "palette_geometry/sphere_rgd_005.png",
                 "palette_geometry/hull_rgd_005.png",
                 "palette_geometry/hull_gauss_rgd.png",
                 "palette_geometry/oklab_hull_rgd.png",
                 "palette_geometry/ours_005.png",
                 "palette_geometry/ours_125.png"],
        luts=[LUT_SRGB_EUCLIDEAN, LUT_CIELAB_EUCLIDEAN, LUT_CIELAB_DELTAE00,
              LUT_CIELAB_SPHERE_RGD_005, LUT_CIELAB_HULL_RGD_005,
              LUT_CIELAB_HULL_GAUSSIAN_RGD_005, LUT_OKLAB_HULL_RGD_005,
              LUT_CIELAB_NEURAL_RGD_005, LUT_CIELAB_NEURAL_RGD_125],
    ),
}

TABLES = {
    # tab_timing goes first: it invalidates + rebuilds its two LUTs from
    # scratch to measure wall-clock construction time, then leaves them
    # cached. Every downstream script that uses those two LUTs (fig_teaser,
    # fig_hue_histograms, tab_scene) then hits cache. Reorder at your peril —
    # reversing it doubles the ~1h ΔE₀₀ rebuild.
    "tab_timing": dict(
        module="scripts.paper.tab_timing",
        outputs=["timing/timing.tex"],
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_125],
    ),
    "tab_scene": dict(
        module="scripts.paper.tab_scene",
        outputs=["scene/scene.tex"],
        luts=[LUT_CIELAB_DELTAE00, LUT_CIELAB_NEURAL_RGD_125],
    ),
}
