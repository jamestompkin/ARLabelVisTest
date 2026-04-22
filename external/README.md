# External code and assets

Everything third-party to the research codebase lives here. Nothing under `external/` is maintained by us — it's either vendored, submoduled, or documented for re-import.

## `matlab_rgd/`

Edelstein et al. 2023 *Regularized Geodesic Distances* — MATLAB ADMM implementation. Called out of band from the Python pipeline via `matlab -batch`; exchanges `.off` meshes and `max_indices_*.txt` files with `arlabelvis/`.

Run the pipeline step from the repo root:

```bash
matlab -batch "cd('external/matlab_rgd'); demo"
```

## `neural_bounding/`

Liu et al. 2024 (SIGGRAPH Conference Papers) — neural bounding networks, used in the "neural" smoothing mode. Git submodule; has its **own** conda environment — do not merge into the main `uv` env.

Initial setup:

```bash
git submodule update --init
cd external/neural_bounding
conda env create -f environment.yml
conda activate neural_bounding
./run.sh
```

## `unity/`

Unity 2021.3.14f1 AR application (URP). Consumes the LUT produced by the Python pipeline and renders the label overlay at runtime. Lives under `external/` because it's a distinct build target, not the research codebase itself. Its own `.gitignore` handles Unity build artifacts (`Library/`, `Temp/`, etc.).

To open in Unity, restore the `unity_fantasy_forest/` asset into `external/unity/Assets/` (see below), then open `external/unity` as the project root.

## `unity_fantasy_forest/`

Unity Asset Store "Fantasy Forest Environment — Free Sample" (TriForge). Scene content for the AR label-overlay demo. Kept separate from `unity/Assets/` so the Unity project directory doesn't visually mix our code with a third-party asset pack.

**To use in Unity**, symlink (preferred) or copy this directory back into `external/unity/Assets/`.

Windows (developer mode or admin), from the repo root:

```
mklink /D "external\unity\Assets\Fantasy Forest Environment Free Sample" "..\..\unity_fantasy_forest"
mklink    "external\unity\Assets\Fantasy Forest Environment Free Sample.meta" "..\..\unity_fantasy_forest.meta"
```

macOS / Linux:

```sh
cd external/unity/Assets
ln -s ../../unity_fantasy_forest "Fantasy Forest Environment Free Sample"
ln -s ../../unity_fantasy_forest.meta "Fantasy Forest Environment Free Sample.meta"
```

`external/unity/.gitignore` already ignores the restored path so the symlink/copy doesn't reintroduce the asset into version control.
