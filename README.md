# ARLabelVisTest

Codebase for the Yang-Maccini senior thesis and the derived IEEE VIS 2026 short paper on smoothing color spaces for AR area-label color selection.

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
