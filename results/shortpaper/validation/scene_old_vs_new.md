# Per-frame label-color gradient: OLD vs NEW for the 'ours' row

Scene: `dubai_changing_bgcolors`. Same processor, same Kwon baseline, same
naive-Gaussian baseline. Only difference is which LUT goes into the
'Ours' row -- production-code OLD bake-out vs NEW orthogonal-axis
bake-out.

| Method | Grad. max | Grad. avg | Grad. median | Grad. std |
|---|---|---|---|---|
| Original (Kwon DeltaE00) | 329.07 | 13.21 | 0.00 | 57.50 |
| Smoothed (3D Gaussian baseline) | 237.55 | 11.41 | 0.00 | 31.75 |
| Ours OLD (production code today) | 361.95 | 6.19 | 0.00 | 43.28 |
| Ours NEW (orthogonal-axis algorithm) | 323.54 | 3.80 | 0.00 | 20.76 |

## Reading the comparison

The paper headline (217.95 -> 41.60 max-gradient, 23.40 -> 6.79 std)
compares the 'Original' row to the 'Ours' row. Under OLD, those
numbers were what the paper reported. Under NEW, the 'Ours NEW' row
is the corresponding measurement, and the headline shifts to
**329.07 -> 323.54 max-gradient, 57.50 -> 20.76 std**.
