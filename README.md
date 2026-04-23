# ARLabelVisTest

Codebase for the Yang-Maccini senior thesis and the derived IEEE VIS 2026 short paper on smoothing color spaces for AR area-label color selection.

## Layout

```
arlabelvis/        # library — methods. Importable, no side effects on import.
  rgd/             #   regularized geodesic distances (Python port of Edelstein 2023, replaces MATLAB)
  colors.py        #   RGB <-> LAB/OKLAB conversions
  distances.py     #   furthest-color search + geodesic-field rendering
  bounding.py      #   sphere / neural / optimized-mesh bounding methods
  meshing.py       #   point-cloud -> triangle-mesh + pytorch mesh optimizer
  voxels.py        #   point-cloud -> binvox
  binvox_rw.py     #   bundled binvox I/O
  off.py           #   OFF file read/write
  interpolate.py   #   sparse LUT -> dense 256^3
  metrics.py       #   numerical metrics + hue/gradient plots
  scene.py         #   27-bin tertile CEC + LUT-lookup per video frame
  viz.py           #   lookup-table renderers (cube, point cloud, hue histogram)

scripts/           # entry points — each generates an artifact, edit CONFIG at the top
  build_mesh.py            # build boundary mesh of the input color space
  smooth_mesh.py           # apply sphere/neural/pytorch smoothing, write OFF
  to_voxels.py             # write binvox for neural_bounding trainer
  run_rgd.py               # consume MATLAB indices -> furthest RGB
  write_lut_files.py       # write sparse candidate LAB + RGB
  interpolate_lut.py       # sparse -> dense LUT text files
  full_pipeline.py         # MATLAB indices -> furthest -> interpolated LUT
  test_matlab.py           # visualize MATLAB geodesic field on the source mesh
  show_furthest_mesh.py    # mesh colored by furthest output RGB
  metrics_report.py        # LUT metrics + alpha-sweep plot
  process_scene_video.py   # Unity replacement: video + masks -> label-color CSV
  render_lut_cube.py       # paper-ready isometric cube from a LUT
  render_hue_histogram.py  # hue distribution of a LUT
  _config.py               # shared RunConfig dataclass + filename conventions

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

Neural bounding submodule has its own environment (do not merge with the main env):

```bash
git submodule update --init
cd external/neural_bounding
conda env create -f environment.yml
conda activate neural_bounding
./run.sh
```

## End-to-end LUT construction

The pipeline is a four-stage handshake — each stage's output feeds the next:

1. **Voxelize the color space**: `uv run python -m scripts.to_voxels` → `external/neural_bounding/data/3D/<space>_<interval>_<dim>.binvox`
2. **Train neural bounder** (in the submodule's conda env): `./run.sh` over that binvox → `data/neural_bounding_<space>_<dim>.binvox`
3. **Build smoothed OFF**: `uv run python -m scripts.smooth_mesh` → `external/matlab_rgd/RGB2<space>_<smoothing>_<interval>.off`
4. **Run all-pairs RGD**: `uv run python -m scripts.run_rgd_python --from-config` → `external/matlab_rgd/max_indices_<...>.txt`
   - Uses [arlabelvis/rgd/admm.py](arlabelvis/rgd/admm.py) (port of Edelstein 2023's `rdg_ADMM.m`).
   - Validated against MATLAB to rel L2 ≤ 5e-14 on a 10K-vertex mesh, 100% argmax agreement on 642-source all-pairs, 99.8% on non-symmetric 2562-source mesh with every mismatch a <5e-15 plateau flip. See `validation/compare_matlab_vs_python.py` (8 regression tiers) and `validation/MATLAB_PORT_PLAN.md`.
   - Fallback A/B reference: `matlab -batch "cd('external/matlab_rgd'); demo"` writes the same file in the same format. Only needed for independent verification.
5. **Assemble + interpolate LUT**: `uv run python -m scripts.full_pipeline` → `AllCandidateLABvals_<...>.txt`

For figures and metrics:

```bash
uv run python -m scripts.render_lut_cube <lab-file> <rgb-file> -o figures/cube.png
uv run python -m scripts.render_hue_histogram <lab-file> <rgb-file> -o figures/hue.png
uv run python -m scripts.metrics_report
```

For the scene side (replaces Unity):

```bash
uv run python -m scripts.process_scene_video \
  --video scene.mp4 --label-mask label.png \
  --lab-file AllCandidateLABvals_*.txt --rgb-file AllCorrespondingRGBVals_*.txt \
  -o data/label_colors_export.csv
```

## Validation

`validation/` contains a known-ground-truth icosphere + analytic geodesics for checking the MATLAB RGD solver:

```bash
uv run python validation/validate_rgd_toy.py                         # writes Python references
"C:/Program Files/MATLAB/R2026a/bin/matlab.exe" -batch "cd('validation'); run_matlab_validation"
uv run python validation/compare_matlab_vs_python.py                 # diff
```
