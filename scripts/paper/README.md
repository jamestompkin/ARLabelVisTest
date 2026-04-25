# Paper reproducibility pipeline

Every figure and table that depends on the pipeline has its own script in this
directory. The pipeline has three output consumers, each with its own output
directory:

| Consumer | Output dir | Registry |
|---|---|---|
| Thesis-style figures | `results/paper/figures/` | `FIGURES` in `_configs.py` |
| Thesis-style tables  | `results/paper/tables/`  | `TABLES` in `_configs.py` |
| IEEE VIS 2026 short paper figures | `SHORTPAPER_FIG_DIR` (Overleaf `ieeevis2026/figures/`) | `SHORTPAPER_FIGURES` |
| IEEE VIS 2026 short paper tables  | `SHORTPAPER_TAB_DIR` (Overleaf `ieeevis2026/tables/`)  | `SHORTPAPER_TABLES`  |

## Layout

- `_lut_cache.py` — content-addressable LUT cache keyed by `LutConfig`
  (`space × smoothing × sigma × metric × alpha_hat × interval × voxel_dim ×
  output_space`). Missing entries are built from scratch by running the
  right pipeline stages: `generate_labs → mesh → farthest-color → interpolate`.
  Cached as `.npy` under `data/luts/cache/`.
- `_configs.py` — registries. Every figure/table declares the `LutConfig`s it
  needs; identical configs from different scripts hit the same cache entry.
- `_shared.py` — `FIG_DIR`, `TAB_DIR`, `SHORTPAPER_FIG_DIR`, `SHORTPAPER_TAB_DIR`,
  and `lut_to_srgb_u8(lut, cfg)` which dispatches on `cfg.output_space`
  (sRGB-valued vs CIELAB-valued LUTs).
- `_shortpaper.py` — `fig_path(subdir, name)` and
  `tab_path(subdir, name)` helpers. Every short-paper output
  lives in its own per-figure/per-table subdirectory so the Overleaf layout
  matches the LaTeX figure/table labels.
- `fig_*.py`, `tab_*.py` — one per thesis-style figure/table.
- `shortpaper_*.py` — one per short-paper figure/table.
- `reproduce_all.py` — iterates every registered script, skips scripts
  whose output files already exist (unless `--force`).
- `_migrate_cache.py` — one-shot migrations for field-rename or value-rename
  refactors that invalidate cache keys. Idempotent; run after any such
  refactor to keep pre-existing cache entries resolvable.

## Running

```bash
# All registered outputs (thesis + tables + short paper), skip existing:
uv run python -m scripts.paper.reproduce_all

# Force rebuild:
uv run python -m scripts.paper.reproduce_all --force

# Just the short paper (Overleaf outputs):
uv run python -m scripts.paper.reproduce_all --shortpaper-only

# Everything except the short paper:
uv run python -m scripts.paper.reproduce_all --skip-shortpaper

# Subset by ID:
uv run python -m scripts.paper.reproduce_all --only fig_hue_histograms tab_geometry_smoothing

# One script directly:
uv run python -m scripts.paper.fig_hue_histograms
uv run python -m scripts.paper.shortpaper_tab_scene
```

The short-paper outputs live outside this repo (in the Overleaf project).
Their location is controlled by `ARLABELVIS_SHORTPAPER_DIR`; default is
James's Dropbox path. Override for a different Overleaf checkout:

```bash
ARLABELVIS_SHORTPAPER_DIR=/path/to/overleaf/ieeevis2026 \
  uv run python -m scripts.paper.reproduce_all --shortpaper-only
```

## Interval (sampling density)

`interval=16` is the default: ~17³ = 4913 input sRGB points, runs in tens of
seconds to a few minutes per LUT. `interval=1` is the thesis-fidelity
setting (full 256³ = 16.7M points) and runs ~hours to a day. Override by
editing the `LutConfig.interval` field where desired.

## Coverage

Thesis-style figures and tables — all working end-to-end:

- ✓ `fig_srgb_euclidean`, `fig_cielab_euclidean`
- ✓ `fig_color_space_comparison` (OKLAB vs CIELAB)
- ✓ `fig_sphere_lookup`, `fig_neural_bounded_lut`
- ✓ `fig_gaussian_smoothing` (3-panel σ sweep)
- ✓ `fig_distance_comparison` (Euclidean / ΔE₇₆ / RGD)
- ✓ `fig_alpha_sweep_cubes` (9-panel α̂ sweep)
- ✓ `fig_alpha_plots` (α̂ vs 4 metrics)
- ✓ `fig_hue_histograms` (Euclidean / ΔE₇₆ / RGD)
- ✓ `tab_space_and_interp`, `tab_distance_metric`, `tab_geometry_smoothing`
  (all with the full metric set: smoothness + color-diversity)
- ✗ `fig_smoothed_original` (naive-smoothed Kwon baseline) — not written; the
  short paper's `tab:scene` already uses this baseline as a LUT.

Short paper (IEEE VIS 2026) outputs:

- ✓ `shortpaper_teaser` → `teaser/{old_lookup,neural_bounding_05}.png`
- ✓ `shortpaper_space_comparison` → `space_comparison/{rgb_euclidean,oklab_rgd_05,cielab_rgd_05}.png`
- ✓ `shortpaper_smoothing_zooms` → `smoothing/{zoom_4o0,zoom_neural}.png`
- ✓ `shortpaper_hue_histograms` → `hue_histograms/{OriginalLABVals_hue,AllCandidateLABvals_CIELAB_1_Euclidean_hue,AllCandidateLABvals_CIELAB_1_RGD_50_neural_256_hue}.png`
- ✓ `shortpaper_alpha_plots` → `alpha_plots/alpha_plots.png`
- ✓ `shortpaper_tab_scene` → `scene/scene.tex` (per-frame gradient stats on Dubai scene)
- ✗ `shortpaper_tab_timing` — LUT build-time table; needs an interval=1 pass to be directly comparable to Kwon's 41-min claim.

`_configs.py::LUT_ALPHA_SWEEP` uses nine α̂ values matching the thesis range.

### Smoothing-path notes

- **sphere**: constructs a 2562-vertex icosphere whose radius matches the
  bounding sphere returned by `bind_lab_to_sphere`. Using a direct icosphere
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

### Cube renderer (PyVista / VTK)

The isometric LUT-cube renderer in `arlabelvis.viz.render_srgb_cube_isometric`
uses PyVista + VTK. Windows Smart App Control can block unsigned VTK DLLs
(the error reads "An Application Control policy has blocked this file") —
this can appear *mid-session* if the reputation feed updates during a run.

Fallback: the same function tries PyVista first and falls through to a
matplotlib `plot_surface`-based renderer if VTK can't load, with a
warning. Output is visually distinguishable but usable for review drafts;
re-run `--force` after unblocking VTK for publication-quality renders.

## Adding a new figure or table

Thesis-style:
1. Add a `LutConfig` (or reuse one) in `_configs.py`.
2. Register the figure/table in the `FIGURES` or `TABLES` dict.
3. Drop a `fig_*.py` / `tab_*.py` script that:
   - calls `get_lut(CONFIG)` to get a dense 256³ LUT,
   - renders / computes metrics,
   - writes to `FIG_DIR` / `TAB_DIR` with a stable filename.

Short paper:
1. Add a `LutConfig` if the figure/table needs a new LUT.
2. Register in `SHORTPAPER_FIGURES` or `SHORTPAPER_TABLES` with output
   paths prefixed `shortpaper:<rel>` (figures) or `shortpaper_tab:<rel>`
   (tables) where `<rel>` is `<subdir>/<file>`.
3. Drop a `shortpaper_*.py` script that writes via
   `fig_path("<subdir>", "<file>.png")` or
   `tab_path("<subdir>", "<file>.tex")`.

`reproduce_all.py` picks up new registry entries automatically.
