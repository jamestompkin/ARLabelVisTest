# Validation plan for RGD_MATLAB on icosphere_sub3.off

## Inputs produced by validate_rgd_toy.py
- `icosphere_sub3.off` — unit sphere mesh, 642 vertices, 1280 triangles (0-indexed on disk per the OFF spec; `readOff.m` adds +1 after parsing)
- `reference_analytic_geodesic.csv` — exact great-circle distance from vertex 0 (MATLAB index 1)
- `reference_cot_laplacian_dense.csv` — Python cotangent Laplacian in the same sign convention as `cotLaplacian.m` (positive diagonal, negative off-diagonals, row-sums zero)

## Tier 1 — unit level (cotLaplacian.m)
Run a one-off MATLAB snippet:
```matlab
Mm = MeshClass('icosphere_sub3');   % drops the .off; MeshClass prepends it
[W_mat, ~] = cotLaplacian(Mm);
writematrix(full(W_mat), 'matlab_cot_laplacian_dense.csv');
```
Then diff against `reference_cot_laplacian_dense.csv`. Acceptance:
- `max |W_py - W_mat|  < 1e-10`  (Frobenius err / nnz ~= machine epsilon)

If this fails, something is wrong at the matrix-assembly level and nothing else can be trusted.

## Tier 2 — toy-mesh RGD (demo.m-style)
Edit a copy of `demo.m` to point at `icosphere_sub3` with source `x0 = 1`, sweep `alpha_hat0` across e.g. `[1e-3, 1e-2, 0.05, 0.25, 1.0]`, and for each run save `u_D1.csv`.

### Checks per alpha_hat
Numerical (no closed form at finite α̂):
- `u[src=1] ~= 0`  (tolerance scales with alpha_hat)
- `u >= 0` everywhere
- `u` is symmetric under source swap: if we also run with source = argmax, u at vertex 1 should equal the original u at argmax (RGD is symmetric)

Convergence to analytic geodesic (as alpha_hat -> 0):
- As alpha_hat decreases, MATLAB `u` should approach `reference_analytic_geodesic.csv` up to an overall scale factor (RGD normalization may differ — check rank-correlation and per-vertex ratio stability rather than raw equality).
- Relative L2 between MATLAB `u/max(u)` and `d_ref/max(d_ref)` should decrease monotonically as alpha_hat -> 0.

### Argmax check
At reasonable alpha_hat (0.05-0.25), `argmax(u)` should be near the antipode of vertex 1.
- Analytic argmax (MATLAB index) = 4
- Tolerance: MATLAB's argmax should be that vertex OR a direct neighbor of it (subdivided icospheres don't always have an exact antipode; expected geodesic to the computed antipode ~= 3.141593 rad).

## Tier 3 — gold standard (deferred)
Run `demo.m` on the actual CIELAB neural-bounded mesh (`external/matlab_rgd/RGB2CIELAB_neural_1.off`) with `alpha_hat = 1.25`, save `u_D1.csv` and `max_indices_*.txt`. Treat as reference for everything downstream. No further port validation needed unless we port.
