"""Farthest-color mapping figure for the LUT comparison.

The figure makes the function ``f: sRGB_in -> sRGB_out`` the visual
subject — what the LUT actually *does to its input*, not what
codomain it spans. Two earlier failure modes informed the current
design:

- An "output palette" panel (palette dots in CIELAB chromaticity)
  was redundant with the partition view and looked sparse and forlorn
  for low-K methods (sRGB+Euclidean had its 8 corner dots scattered
  around an empty axis). Removed.
- The dense LUT was being filled by trilinear interpolation between
  algorithmic picks, so the on-disk LUT contained thousands of blends
  that aren't farthest from anything. ``LutConfig.interp_method``
  defaults to ``'nearest'`` now, so the dense LUT and the palette
  extracted from it both reflect only the algorithmic picks.

Per method:

- The **per-method PNG** has 3 panels (3D + 2D partition + pushforward),
  because there's room.
- The **combined PNG** stacks 2 panels per row (2D partition +
  pushforward) for legibility at the dense thumbnail scale.

Panel semantics:

  1. INPUT PARTITION (a*, b*) — every visible dot is an *input* voxel
     plotted at its own CIELAB position, coloured by the LUT's output
     sRGB. Solid colour blobs reveal the partition: sRGB+Euclidean
     splits the gamut into 8 octants of one corner each, neural+RGD
     into ~100 small cells. The colours visible in this panel ARE the
     palette.
  2. INPUT PARTITION (3D CIELAB) — same data with L* on the third
     axis. Per-method PNG only; dropped from the combined panel for
     readability.
  3. PUSHFORWARD — sorted-by-mass distribution over palette entries,
     log-log. Grey fill traces the long tail; coloured bars mark the
     top-K_eff entries (the "effective palette"). The dashed vertical
     line is K_eff.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from arlabelvis.colors import srgb_to_lab
from arlabelvis.luts import get_lut, lut_to_srgb_u8
from scripts.paper._configs import (LUT_CIELAB_DELTAE00,
                                    LUT_CIELAB_EUCLIDEAN,
                                    LUT_CIELAB_HULL_GAUSSIAN_RGD_005,
                                    LUT_CIELAB_HULL_RGD_005,
                                    LUT_CIELAB_NEURAL_RGD_005,
                                    LUT_CIELAB_NEURAL_RGD_125,
                                    LUT_CIELAB_SPHERE_RGD_005,
                                    LUT_OKLAB_HULL_RGD_005,
                                    LUT_SRGB_EUCLIDEAN)
from scripts.paper._paths import fig_path


def _palette_from_lut(lut_u8: np.ndarray, *,
                       interval: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(palette_rgb_u8, counts)`` for the *algorithmic* output set,
    sampled at the LUT's seed grid (not the dense bake-out).

    The dense 256^3 LUT is built by trilinear-interpolating between the
    farthest-colour picks at the seed grid. For this paper's purposes the
    interpolated voxels are noise — they're blends of two different
    algorithmic picks, so they don't correspond to "the farthest colour
    from anything"; they're whatever sRGB triplet falls between two
    unrelated palette entries. Including them makes a method like
    sRGB+Euclidean (8 algorithmic picks) appear to have K=4913 unique
    outputs, which is misleading.

    Sampling at ``generate_input_grid('sRGB', interval)`` positions
    returns exactly the picks the farthest-point operator made, with
    counts equal to the number of seeds that picked each one. Bit-packs
    (R, G, B) into a 24-bit int so the unique step is ~50× faster than
    ``np.unique(axis=0)``.
    """
    rgb_idx, _ = _input_subsample(interval=interval, max_dots=10**9)
    seeds = lut_u8[rgb_idx[:, 0], rgb_idx[:, 1], rgb_idx[:, 2]].astype(np.uint32)
    packed = (seeds[:, 0] << 16) | (seeds[:, 1] << 8) | seeds[:, 2]
    keys, counts = np.unique(packed, return_counts=True)
    palette = np.stack([(keys >> 16) & 0xFF,
                        (keys >> 8) & 0xFF,
                        keys & 0xFF], axis=1).astype(np.uint8)
    order = np.argsort(-counts)
    return palette[order], counts[order]


def _rgb_u8_to_cielab(rgb_u8: np.ndarray) -> np.ndarray:
    return srgb_to_lab(rgb_u8.astype(np.float64))


def _input_subsample(interval: int = 16, max_dots: int = 1500
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb_idx, cielab_pos)`` aligned to the LUT's *seed* grid.

    The dense 256^3 LUT is built by trilinear-interpolating between seeds
    sampled at ``generate_input_grid('sRGB', interval)`` positions. Off
    the seed grid the LUT is a continuous blend of neighbouring seeds'
    output sRGBs — visually that smears the partition. Sampling here at
    the seed positions exactly returns the algorithm's discrete
    farthest-colour assignments, which is what the panels are about.

    For ``interval=16`` the seed grid is 17 points per axis at
    ``[0, 15, 31, 47, ..., 239, 255]`` (note the -1 shift in
    ``generate_input_grid``); 17^3 = 4913 voxels. To keep matplotlib's
    3D scatter renderer from hitting many-GB allocations on the
    9-method combined figure, we subsample uniformly to ``max_dots``
    (default 1500). The subsample seed is fixed so the same indices
    are used for every method — visual comparisons remain honest.
    """
    cached = _INPUT_SUBSAMPLE_CACHE.get((interval, max_dots))
    if cached is not None:
        return cached
    from arlabelvis.meshing import generate_input_grid
    rgb_idx, _ = generate_input_grid("sRGB", interval=interval)
    if len(rgb_idx) > max_dots:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(rgb_idx), size=max_dots, replace=False)
        rgb_idx = rgb_idx[idx]
    cielab = srgb_to_lab(rgb_idx.astype(np.float64))
    _INPUT_SUBSAMPLE_CACHE[(interval, max_dots)] = (rgb_idx, cielab)
    return rgb_idx, cielab


_INPUT_SUBSAMPLE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _size_from_counts(counts: np.ndarray,
                      s_min: float = 2.0, s_max: float = 140.0) -> np.ndarray:
    """Log-scaled marker sizes. Rare entries ~2px, dominant ~140px."""
    logc = np.log10(counts.astype(float) + 1.0)
    lo, hi = logc.min(), logc.max()
    if hi <= lo:
        return np.full_like(counts, (s_min + s_max) / 2, dtype=float)
    t = (logc - lo) / (hi - lo)
    return s_min + (s_max - s_min) * t


def _effective_palette_size(counts: np.ndarray) -> float:
    """Inverse participation ratio of the pushforward, ``K_eff = 1/Σ p_i²``.

    Equals K for uniform distributions, collapses toward 1 for spiked ones.
    """
    p = counts.astype(np.float64) / counts.sum()
    return float(1.0 / np.sum(p * p))


def _render_partition_3d(ax, lut_u8: np.ndarray, *,
                          marker_size: float = 10.0) -> None:
    """Input voxels in 3D CIELAB, coloured by their LUT-assigned output sRGB.

    This is the figure's headline panel: solid blocks of one colour reveal
    contiguous regions of the input space all mapped to the same farthest
    colour. The shape of those blocks distinguishes "8-octant naive
    Euclidean" from "100-cell neural+RGD" at a glance.
    """
    rgb_idx, cielab = _input_subsample()
    out_rgb = lut_u8[rgb_idx[:, 0], rgb_idx[:, 1], rgb_idx[:, 2]]
    ax.scatter(cielab[:, 1], cielab[:, 2], cielab[:, 0],
               c=out_rgb.astype(np.float32) / 255.0,
               s=marker_size, edgecolors="none", alpha=0.9)
    ax.set_xlabel("input a*"); ax.set_ylabel("input b*"); ax.set_zlabel("input L*")
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(0, 100)
    ax.view_init(elev=20, azim=-55)
    ax.set_title("input partition (3D)", fontsize=9)


def _render_partition_ab(ax, lut_u8: np.ndarray, *,
                          marker_size: float = 10.0) -> None:
    """Input voxels in 2D (a*, b*) chromaticity, coloured by output sRGB.

    Counterpart to the 3D partition without L*-axis occlusion. Projecting
    pulls every L* slice down onto the same plane, which is exactly what
    makes the partition cells visible as solid blobs of colour.
    """
    rgb_idx, cielab = _input_subsample()
    out_rgb = lut_u8[rgb_idx[:, 0], rgb_idx[:, 1], rgb_idx[:, 2]]
    ax.scatter(cielab[:, 1], cielab[:, 2],
               c=out_rgb.astype(np.float32) / 255.0,
               s=marker_size, edgecolors="none", alpha=0.7)
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("input a*"); ax.set_ylabel("input b*")
    ax.set_title("input partition (a*, b*)", fontsize=9)


def _render_pushforward(ax, palette_rgb: np.ndarray, counts: np.ndarray,
                         K: int, K_eff: float, hist_x_max: int) -> None:
    """Sorted pushforward distribution, log-log axes.

    Per-entry coloured vertical lines (vlines) — line *thickness* is in
    points, so it doesn't change as log-x compresses the axis. Log-x
    therefore compresses position only, not visual width. At high K
    the vlines overlap into a coloured streak, which honestly conveys
    "many entries, low mass each." Each line is coloured by the
    corresponding palette entry's sRGB so the visual link to the
    partition panel is preserved.

    Dashed vertical line marks ``K_eff``.
    """
    marker_colors = palette_rgb.astype(np.float32) / 255.0
    frac = counts / counts.sum()
    ranks = np.arange(1, len(frac) + 1)
    ax.vlines(ranks, ymin=1e-9, ymax=frac,
              colors=marker_colors, linewidth=1.0)
    ax.axvline(K_eff, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, max(hist_x_max, 10))
    ax.set_ylim(1e-7, 1.0)
    ax.set_xlabel("palette entry (rank, log)")
    ax.set_ylabel("voxel fraction (log)")
    ax.set_title(f"pushforward (K = {K},  $K_{{\\mathrm{{eff}}}}$ = {K_eff:.0f})",
                  fontsize=9)


def _render_method_full(axes: tuple, lut_u8: np.ndarray, title: str,
                         hist_x_max: int) -> None:
    """Per-method 3-panel render: ``(ax_part_3d, ax_part_ab, ax_hist)``."""
    ax_part_3d, ax_part_ab, ax_hist = axes
    palette_rgb, counts = _palette_from_lut(lut_u8)
    K = len(palette_rgb)
    K_eff = _effective_palette_size(counts)

    _render_partition_3d(ax_part_3d, lut_u8)
    ax_part_3d.set_title(f"{title}\ninput partition (3D)  K={K}",
                          fontsize=9)
    _render_partition_ab(ax_part_ab, lut_u8)
    ax_part_ab.set_title(f"input partition (a*, b*)  $K_{{\\mathrm{{eff}}}}$={K_eff:.0f}",
                          fontsize=9)
    _render_pushforward(ax_hist, palette_rgb, counts, K, K_eff, hist_x_max)


def _render_method_compact(axes: tuple, lut_u8: np.ndarray, title: str,
                            hist_x_max: int) -> None:
    """Combined-figure 2-panel render: ``(ax_part_ab, ax_hist)``."""
    ax_part_ab, ax_hist = axes
    palette_rgb, counts = _palette_from_lut(lut_u8)
    K = len(palette_rgb)
    K_eff = _effective_palette_size(counts)

    _render_partition_ab(ax_part_ab, lut_u8)
    ax_part_ab.set_title(
        f"{title}\ninput partition (a*, b*)   K={K}  $K_{{\\mathrm{{eff}}}}$={K_eff:.0f}",
        fontsize=9,
    )
    _render_pushforward(ax_hist, palette_rgb, counts, K, K_eff, hist_x_max)


def _render_single(cfg, title, slug, hist_x_max: int) -> Path:
    """Per-method 3-panel figure: input partition (3D + 2D) + pushforward."""
    lut_u8 = lut_to_srgb_u8(get_lut(cfg), cfg.output_space)
    fig = plt.figure(figsize=(11.0, 3.3), dpi=96)
    ax_part_3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax_part_ab = fig.add_subplot(1, 3, 2)
    ax_hist = fig.add_subplot(1, 3, 3)
    _render_method_full((ax_part_3d, ax_part_ab, ax_hist),
                         lut_u8, title, hist_x_max=hist_x_max)
    fig.tight_layout()
    out = fig_path("palette_geometry", f"{slug}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out


def _format_title(cfg) -> str:
    """Four-slot spec used as every row's title.

    Format: ``<input> -> <working> | <shape> | <smoothing> | <metric>``

    Slots are always present so rows line up visually for cross-row
    comparison: ``smoothing="none"`` renders as ``--`` rather than
    being omitted, and metrics reduce to canonical strings (``L2`` for
    Euclidean, ``ΔE₀₀`` rendered via mathtext, ``RGD α̂=N`` for RGD).
    ASCII separators only — the figure's matplotlib backend handles
    Unicode fine, but the script's stdout under Windows ``charmap``
    does not, and we need the title to be the same string in both
    places.
    """
    if cfg.smoothing == "none":
        smoothing = "--"
    elif cfg.smoothing == "gaussian":
        smoothing = rf"gauss $\sigma={cfg.shape_gaussian_sigma:g}$"
    else:
        smoothing = cfg.smoothing
    if cfg.metric == "Euclidean":
        metric = "L2"
    elif cfg.metric == "DeltaE76":
        metric = r"$\Delta E_{76}$"
    elif cfg.metric == "DeltaE94":
        metric = r"$\Delta E_{94}$"
    elif cfg.metric == "DeltaE00":
        metric = r"$\Delta E_{00}$"
    elif cfg.metric == "RGD":
        metric = rf"RGD $\hat\alpha={cfg.metric_rgd_alpha_hat:g}$"
    else:
        metric = cfg.metric
    return (f"{cfg.input_space} -> {cfg.working_space}"
            f"  |  {cfg.shape}  |  {smoothing}  |  {metric}")


def main() -> None:
    methods = [
        (LUT_SRGB_EUCLIDEAN,            "srgb_euclidean"),
        (LUT_CIELAB_EUCLIDEAN,          "cielab_euclidean"),
        (LUT_CIELAB_DELTAE00,           "kwon_de00"),
        (LUT_CIELAB_SPHERE_RGD_005,     "sphere_rgd_005"),
        (LUT_CIELAB_HULL_RGD_005,       "hull_rgd_005"),
        (LUT_CIELAB_HULL_GAUSSIAN_RGD_005, "hull_gauss_rgd"),
        (LUT_OKLAB_HULL_RGD_005,        "oklab_hull_rgd"),
        (LUT_CIELAB_NEURAL_RGD_005,     "ours_005"),
        (LUT_CIELAB_NEURAL_RGD_125,     "ours_125"),
    ]
    # Build (cfg, title, slug) triples. Titles come from one consistent
    # formatter so rows are directly comparable.
    methods = [(cfg, _format_title(cfg), slug) for cfg, slug in methods]

    # Pre-pass: determine the shared histogram x-axis range so bar widths are
    # physically comparable across methods. K is computed from the
    # *algorithmic* palette (seed-grid picks, no trilinear bake-out
    # blending) — see ``_palette_from_lut`` for why.
    cached: list[np.ndarray] = []
    ks: list[int] = []
    for cfg, title, _slug in methods:
        print(f"[palette] loading {title}...", flush=True)
        lut_u8 = lut_to_srgb_u8(get_lut(cfg), cfg.output_space)
        cached.append(lut_u8)
        palette, _ = _palette_from_lut(lut_u8)
        ks.append(len(palette))
    hist_x_max = max(ks)
    print(f"[palette] K (algorithmic) per method = {ks}; "
          f"shared hist x-max = {hist_x_max}", flush=True)

    # Per-method panels at readable size.
    for (cfg, title, slug), lut_u8 in zip(methods, cached):
        out = _render_single(cfg, title, slug, hist_x_max)
        print(f"  wrote {out.name}", flush=True)

    # Combined stacked figure for LaTeX inclusion. Two panels per method
    # (partition-ab, pushforward) — drops both the 3D partition (3D
    # plots become unreadable at thumbnail row height, and matplotlib's
    # 3D backend retains buffers across many axes) and the redundant
    # output-palette panel (the colours visible in the partition panel
    # ARE the palette). dpi=75 keeps the saved PNG under 2000 px on the
    # longest edge per the CLAUDE.md image-size cap.
    n = len(methods)
    cols = 2
    fig = plt.figure(figsize=(9.0, 2.8 * n), dpi=75)
    for i, ((cfg, title, _slug), lut_u8) in enumerate(zip(methods, cached)):
        ax_part_ab = fig.add_subplot(n, cols, i * cols + 1)
        ax_hist = fig.add_subplot(n, cols, i * cols + 2)
        _render_method_compact((ax_part_ab, ax_hist), lut_u8, title,
                                hist_x_max=hist_x_max)
    fig.tight_layout()
    combined = fig_path("palette_geometry", "palette_geometry.png")
    fig.savefig(combined, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {combined}", flush=True)


if __name__ == "__main__":
    main()
