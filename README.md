# ARLabelVisTest

Codebase for the Yang-Maccini senior thesis and the derived IEEE VIS 2026 short paper on smoothing color spaces for AR area-label color selection.

## What the algorithm does

For every input voxel `c` in sRGB, we want to pick a *farthest* output color `c' = h(c)` so that overlaid labels stay visible against `c`. The whole paper is one operator:

```
h(c) = argmax_{c' ∈ G} Δ(c, c')
```

where `G` is a **candidate set** of colors and `Δ` is a **distance measure**. The trick is that naïve choices of either make `h` discontinuous in `c` — adjacent voxels pick wildly different farthest colors, which causes flicker on a moving label. Smoothing `h` is the paper's contribution.

### The pipeline

Every method in the paper is one configuration of this five-stage chain:

```
                   ┌──────────────────────┐
sRGB seed grid ──▶ │ 1. generate_input_   │ ──▶ input points + their sRGBs
  (interval=16)    │    grid              │     (e.g. 17³ = 4913 voxels)
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ 2. to_working_space  │ ──▶ same points, in CIELAB / OKLAB / sRGB
                   │   (per cfg.working)  │     (the "perceptually uniform" frame)
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ 3. build_candidates  │ ──▶ CandidateSet(verts, faces, rgbs)
                   │   geometry: hull /   │     - vertices in working space
                   │   sphere / neural    │     - triangulated mesh
                   │   smoothing: none /  │     - rgbs = nearest displayable sRGB
                   │   gaussian (σ)       │       per vertex
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ 4. score_argmax      │ ──▶ per-input candidate index
                   │   metric: Euclidean  │     (which vertex is "farthest")
                   │   ΔE₀₀ / RGD (α̂)     │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ 5. render_candidate  │ ──▶ per-input output color
                   │   (lookup in rgbs    │     in cfg.output_space
                   │    or convert verts) │
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ dense_lut_from_      │ ──▶ 256³ × 3 sRGB-indexed LUT
                   │ sparse (NN-fill)     │     — the AR app's lookup table
                   └──────────────────────┘
```

Every stage lives in [arlabelvis/luts.py](arlabelvis/luts.py); each is a function that takes the specific fields it needs, not a full `LutConfig`, so they're reusable from notebooks.

### The orthogonal axes (a.k.a. "the knobs")

The algorithm is naturally **three orthogonal axes**:

1. **Working color space** — `sRGB` / `CIELAB` / `OKLAB`. Determines what "distance" means.
2. **Candidate-set geometry** — *shape* of the mesh of candidate colors:
   - `sphere` — icosphere circumscribing the gamut. Knob: `radius`.
   - `hull` + `gaussian` smoothing — voxelize the gamut, blur, marching-cubes the level set. Knob: `shape_gaussian_sigma` (σ).
   - `neural` — small ReLU MLP trained on the gamut indicator; mesh = MLP level set. Knob: `shape_neural_bound_bias` (Liu 2024 α/β flip).
3. **Distance measure** — what `Δ` evaluates. Independent of the geometry:
   - `Euclidean` — L2 in working space.
   - `DeltaE00` — perceptual distance, CIELAB only.
   - `RGD` — regularized geodesic distance on the candidate mesh (Edelstein 2023). Knob: `metric_rgd_alpha_hat` (α̂).

There is also a **hidden fourth axis: substrate density** — the face count of the candidate mesh. RGD and Gaussian-on-mesh need the mesh to be dense enough that mean-edge-length is much smaller than the smoothing scale. The bare convex hull at ~166 vertices is too coarse for either to do anything; results on it look like Euclidean. Sphere uses subdivisions=4 → 5120 faces; hull-Gaussian and neural decimate to 2000. This inconsistency is real; see the animation scripts for a clean substrate-density-controlled comparison.

### The named conditions (paper configs)

Defined in [scripts/paper/_configs.py](scripts/paper/_configs.py). Every figure/table in the paper consumes one or more of these:

| Name | Working | Geometry | Smoothing | Metric | α̂ | Role in paper |
|---|---|---|---|---|---|---|
| `LUT_SRGB_EUCLIDEAN` | sRGB | hull | — | Euclidean | — | Naïve baseline (8 cube corners) |
| `LUT_CIELAB_EUCLIDEAN` | CIELAB | hull | — | Euclidean | — | Mid-point in space-comparison |
| `LUT_CIELAB_DELTAE00` | CIELAB | hull | — | ΔE₀₀ | — | Kwon 2019 baseline |
| `LUT_CIELAB_HULL_RGD_005` | CIELAB | hull | — | RGD | 0.05 | Hull + RGD (degenerate — substrate too coarse) |
| `LUT_OKLAB_HULL_RGD_005` | OKLAB | hull | — | RGD | 0.05 | OKLAB variant for space comparison |
| `LUT_CIELAB_SPHERE_RGD_005` | CIELAB | sphere | — | RGD | 0.05 | Outer-bound sphere |
| `LUT_CIELAB_HULL_GAUSSIAN_RGD_005` | CIELAB | hull | gaussian σ=2 | RGD | 0.05 | Smoothed-boundary baseline |
| `LUT_CIELAB_NEURAL_RGD_005` | CIELAB | neural | — | RGD | 0.05 | Ours, low α̂ |
| `LUT_CIELAB_NEURAL_RGD_050` | CIELAB | neural | — | RGD | 0.5 | Ours, mid α̂ (hue-histogram demo) |
| `LUT_CIELAB_NEURAL_RGD_125` | CIELAB | neural | — | RGD | 1.25 | **Final recipe** (teaser, scene table, timing) |
| `LUT_ALPHA_SWEEP` | CIELAB | neural | — | RGD | 0.05–1.5 | α̂ sweep for fig_alpha_plots |
| `LUT_CIELAB_GAUSSIAN_SIGMAS` | CIELAB | hull | gaussian σ∈{0.25,2,4} | RGD | 0.05 | σ sweep for fig_smoothing_zooms |

### Animations: each knob, in isolation

Four scripts under [scripts/paper/](scripts/paper/) animate one knob at a time, holding the rest fixed. Each produces an mp4 with three panels: 3D mesh + 2D input partition (a*, b*) + pushforward histogram. They're standalone supplementary demos, not part of `paper.tex`:

| Script | Sweeps | Range | Metric | Cost (60 frames) |
|---|---|---|---|---|
| [animate_sphere_radius.py](scripts/paper/animate_sphere_radius.py) | sphere `radius` | [20, 140] | Euclidean | ~1 min |
| [animate_alpha_hull.py](scripts/paper/animate_alpha_hull.py) | RGD `α̂` on subdivided hull | [0.01, 2.0] log | RGD | ~2 hrs* |
| [animate_gaussian_hull.py](scripts/paper/animate_gaussian_hull.py) | hull `σ` (Gaussian) | [0, 8] | Euclidean | ~10 min |
| [animate_neural_bias.py](scripts/paper/animate_neural_bias.py) | `bound_bias` on neural | [−3, +3] | Euclidean | ~100 min** |

\* RGD all-pairs ADMM × 60 α̂ values × 2626 hull vertices. The script uses a long-lived multiprocessing pool to amortize Windows worker-spawn cost — don't refactor it back to per-frame `compute_all_pairs_argmax` calls.
\** Retrains the MLP from scratch per frame (20k iterations each).

The `--mp4` flag selects ffmpeg over Pillow GIF (smaller files, no quality loss). Defaults are chosen for paper-quality renders; pass `--frames N` to trade smoothness for speed.

### A warning about the bare convex hull

`LUT_CIELAB_HULL_RGD_005` (and the equivalent OKLAB / Euclidean baselines on `shape="hull"` with no smoothing) builds candidates from the raw 166-vertex hull. RGD on that mesh ≈ Euclidean because the regulariser has no interior vertices to route geodesics around. This is fine for *Euclidean* and *ΔE₀₀* baselines but the "+ RGD" pairing is essentially mislabelled. The α̂ animation deliberately runs on a subdivided hull (5248 faces) so RGD's smoothing actually shows up.

## Layout

```
arlabelvis/        # library — methods. Importable, no side effects on import.
  rgd/             #   regularized geodesic distances (Python port of Edelstein 2023)
  colors.py        #   sRGB <-> CIELAB / OKLAB conversions + ``convert_color`` pivot
  distances.py     #   pairwise color metrics (Euclidean, ΔE76, ΔE94, ΔE00)
  gamut.py         #   ``bind_lab_to_sphere`` for the sphere candidate path
  meshing.py       #   ``generate_input_grid`` + Gaussian-smoothed boundary mesh
  neural_bounding.py # in-process ReLU MLP gamut bounder + exact polytope mesh
  off.py           #   OFF file read/write
  interpolate.py   #   sparse LUT -> dense 256^3 (regular sRGB grid path)
  luts.py          #   ``LutConfig``, ``LutCache``, ``build_lut`` / ``get_lut``
  metrics.py       #   numerical metrics + hue/gradient plots
  cec.py           #   27-bin tertile CEC for video frames
  scene_catalog.py #   manifest-driven scene loader
  scene_video.py   #   per-frame CEC + LUT-lookup pipeline
  viz.py           #   lookup-table renderers (cube, point cloud, hue histogram)
  voxels.py        #   DEPRECATED — sRGB-grid -> binvox (retired binvox path)
  binvox_rw.py     #   DEPRECATED — bundled binvox I/O (retired binvox path)

scripts/
  paper/                   # paper pipeline — content-addressable LUT cache +
                           #   figure/table scripts. See scripts/paper/README.md.
                           #   Entry point: scripts.paper.reproduce_all.
  e2e_demo.py              # minute-long sanity check (get_lut -> cube render)
  to_voxels.py             # DEPRECATED — fed the retired binvox-trained bounder

tests/             # smoke tests; run `uv run python -m tests.<name>`
validation/        # MATLAB <-> Python RGD correctness (icosphere ground truth)
data/              # inputs + intermediate artifacts (*.txt, *.csv, *.binvox)
results/           # paper figures output
external/                   # everything third-party — see external/README.md
  matlab_rgd/               #   Edelstein 2023 RGD-ADMM (MATLAB)
  neural_bounding/          #   git submodule: Liu 2024 (SIGGRAPH)
  unity/                    #   Unity 2021.3.14f1 AR app (label overlay target)
  unity_fantasy_forest/     #   Unity Asset Store sample used by the AR build
```

## Setup

```bash
uv sync                           # Python env for arlabelvis
```

CPU PyTorch by default. For CUDA:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

MATLAB is **not required**. The Python port of Edelstein 2023 RGD in [arlabelvis/rgd/](arlabelvis/rgd/) is byte-compatible with the original MATLAB and replaces it on the critical path. `external/matlab_rgd/` is retained only as an independent reference implementation for A/B comparison. Install MATLAB only if you want to re-run that comparison.

The neural-bounding step is now in-process — a small ReLU MLP trained
inside [arlabelvis/neural_bounding.py](arlabelvis/neural_bounding.py)
whose level set is extracted *exactly* by clipping each polytope of the
ReLU arrangement against the level-set plane. The
`external/neural_bounding/` submodule and its conda environment are no
longer required by the paper pipeline; they are kept only as the
reference implementation that this in-process trainer matches.

## End-to-end LUT construction

The paper pipeline is one call:

```python
from arlabelvis.luts import LutConfig, get_lut
lut = get_lut(LutConfig(working_space="CIELAB", shape="neural", metric="RGD",
                        metric_rgd_alpha_hat=1.25, interval=1,
                        output_space="sRGB"))
```

Internally: ``generate_input_grid → to_working_space → build_candidates →
score_argmax → render_candidate → dense_lut_from_sparse``. Every method in
the paper is a configuration of that pipeline (see
[scripts/paper/_configs.py](scripts/paper/_configs.py) for the named
configs). Results are content-addressable-cached under ``data/luts/cache/``.

All paper figures and tables regenerate via:

```bash
uv run python -m scripts.paper.reproduce_all                      # everything
uv run python -m scripts.paper.reproduce_all --shortpaper-only    # VIS 2026 short paper
```

Minute-long sanity check that the whole chain works (no neural, no MATLAB):

```bash
uv run python -m scripts.e2e_demo
```

Scene-side evaluation (per-frame CEC + LUT lookup over a video, replaces
Unity): call ``arlabelvis.scene_video.process_scene_video(...)`` directly
with an in-memory LUT from ``get_lut``. See
[scripts/paper/tab_scene.py](scripts/paper/tab_scene.py) for the
canonical usage.

## Validation

`validation/` contains a known-ground-truth icosphere + analytic geodesics for checking the MATLAB RGD solver:

```bash
uv run python validation/validate_rgd_toy.py                         # writes Python references
"C:/Program Files/MATLAB/R2026a/bin/matlab.exe" -batch "cd('validation'); run_matlab_validation"
uv run python validation/compare_matlab_vs_python.py                 # diff
```
