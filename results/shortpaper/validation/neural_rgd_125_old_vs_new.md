# LUT_CIELAB_NEURAL_RGD_125 -- OLD vs NEW algorithm

Baked under both algorithms at the full 256^3 dense LUT resolution.
NEW implements the orthogonal-axis framing: admit MLP-inside gamut
samples, restrict argmax to admitted-mapped mesh verts, output is
the closest admitted gamut sample whose mesh vert was the argmax.

## Mesh and admission

- Vertices: 14,741
- Faces:    29,486
- Admitted gamut voxels: 15,173,184 / 16,777,216 (90.4%)
- Mesh verts that received at least one admitted voxel: 14,563 / 14,741

## Per-voxel comparison (NEW vs OLD on the dense 256^3 LUT)

- Voxels where output differs: **16,777,216 / 16,777,216** (100.00%)
- Mean per-voxel L1 sRGB diff: 59.04
- Max per-voxel L1 sRGB diff:  722

## Gradient metrics on the dense LUT

| Algorithm | max-gradient | std |
|---|---|---|
| OLD | 594.96 | 33.88 |
| NEW | 638.09 | 28.33 |
| delta (NEW - OLD) | +43.13 | -5.55 |

## Paper headline (for reference)

The thesis abstract reports max-gradient 217.95 -> 41.60 and std
23.40 -> 6.79 for the same recipe (Kwon's DeltaE00 baseline ->
ours: neural+RGD alpha-hat=1.25). Those numbers came from the OLD
bake-out. Under NEW the 'ours' entry would be (638.09, 28.33).

## Files

- `neural_rgd_125_new_lut.npy`     -- NEW dense LUT (256^3 x 3 uint8)
- `neural_rgd_125_checkpoint.npz`  -- mesh + per-vert RGD argmax + admission
- OLD dense LUT lives in the LutCache under key `fa333a8e75e66829`
