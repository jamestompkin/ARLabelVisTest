# Paper reproducibility pipeline

Every figure and table that depends on the pipeline has its own script in this
directory. Outputs land under `results/paper/figures/` and
`results/paper/tables/` and can be directly included from the Overleaf project
(symlink or copy).

## Layout

- `_lut_cache.py` — content-addressable LUT cache keyed by `LutConfig`
  (color space × smoothing × distance × α̂ × interval × voxel_dim). Missing
  entries are built from scratch by running the right pipeline stages:
  `generate_LABs → mesh → farthest-color → interpolate`. Cached as `.npy`
  under `data/luts/cache/`.
- `_configs.py` — registry. Every figure/table declares the `LutConfig`s it
  needs; identical configs from different scripts hit the same cache entry.
- `_shared.py` — `FIG_DIR`, `TAB_DIR`, and helpers for converting cached
  LUTs (stored as interpolated LAB values) back to uint8 sRGB for rendering.
- `fig_*.py`, `tab_*.py` — one per figure/table, each with a `main()` entry.
- `reproduce_all.py` — iterates every registered script in dependency order,
  skips scripts whose output files already exist (unless `--force`).

## Running

```bash
# All figures + tables (skips what's already on disk):
uv run python -m scripts.paper.reproduce_all

# Force rebuild:
uv run python -m scripts.paper.reproduce_all --force

# Subset:
uv run python -m scripts.paper.reproduce_all --only fig_hue_histograms tab_geometry_smoothing

# One script directly:
uv run python -m scripts.paper.fig_hue_histograms
```

## Current scope

`interval=16` is the default; fast (tens of seconds to a few minutes per LUT).
`interval=1` is the thesis-fidelity setting and runs ~hours to a day. Override
by editing the `LutConfig.interval` field where desired, or (cleaner) adding a
`--interval` CLI flag.

Coverage of thesis figures/tables — all working end-to-end:

- ✓ `fig_srgb_euclidean`, `fig_cielab_euclidean`
- ✓ `fig_color_space_comparison` (OKLAB vs CIELAB)
- ✓ `fig_sphere_lookup`, `fig_neural_bounded_lut`
- ✓ `fig_gaussian_smoothing` (3-panel σ sweep)
- ✓ `fig_distance_comparison` (Euclidean / ΔE₇₆ / RGD)
- ✓ `fig_alpha_sweep_cubes` (9-panel α̂ sweep)
- ✓ `fig_alpha_plots` (α̂ vs 4 metrics)
- ✓ `fig_hue_histograms` (Euclidean / ΔE₇₆ / RGD)
- ✓ `tab_color_space`, `tab_distance_measure`, `tab_geometry_smoothing`
  (all with the full metric set: smoothness + color-diversity)
- ✗ `fig_smoothed_original` (naive-smoothed Kwon baseline) — TODO
- ✗ `tab_label_comparison` (scene-frame gradient stats) — needs a shared
  test video; TODO once that's checked in
- ✗ `fig_first_frames` / `fig_second_frames` (frame sequences) — same test
  video dependency

`_configs.py::LUT_ALPHA_SWEEP` uses nine α̂ values matching the thesis range.

### Smoothing-path notes

- **sphere**: constructs a 2562-vertex icosphere whose radius matches the
  bounding sphere returned by `bindLABtoSphere`. Using a direct icosphere
  rather than the hull of sphere-mapped grid points keeps the mesh uniform
  and tractable for RGD at any `interval`.
- **gaussian**: always builds the voxel mesh from the full 256³ sRGB cube
  (independent of `cfg.interval`), because small σ values require fully-
  populated voxels to avoid fragmenting into disconnected components. The
  output mesh is post-decimated to ~2000 faces before RGD all-pairs.
- **convex_hull**: direct hull of the `interval`-sampled LAB points.
  Fast; used as the "unsmoothed gamut mesh" row in the distance-measure
  and geometry-smoothing tables.
- **neural**: reads `data/neural_bounding_<SPACE>_<DIM>.binvox`, extracts
  its surface and decimates to ~2000 faces.

## Adding a new figure

1. Add a `LutConfig` (or reuse one) in `_configs.py`.
2. Register the figure in the `FIGURES` or `TABLES` dict there.
3. Drop a `fig_*.py` / `tab_*.py` next to the others that:
   - calls `get_lut(CONFIG)` to get a dense 256³ LUT,
   - renders or computes metrics,
   - writes to `FIG_DIR` / `TAB_DIR` with a stable filename.
4. `reproduce_all.py` picks it up automatically.
