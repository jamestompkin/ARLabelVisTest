# Removing MATLAB from the RGD pipeline

> **Status: DONE.** Python port is the default in [README.md](../README.md) and
> [scripts/run_rgd_python.py](../scripts/run_rgd_python.py). MATLAB is
> reference-only. All 8 regression tiers in
> [validation/compare_matlab_vs_python.py](compare_matlab_vs_python.py) pass:
> operator-level diffs at ≤5e-15, full solver at ≤5e-14 rel L2 on 10K-vertex
> meshes, 99.8% exact argmax on 2562-source non-symmetric all-pairs (with every
> mismatch a <5e-15 plateau flip), all three regularizers (`D`, `vfa`, `H`), and
> `smooth_vf` matching MATLAB to rel-Fro 5.8e-12. `curved_hessian` (Stein 2020)
> is provided via `igl.curved_hessian_energy` from libigl — same algorithm, same
> author.

---

Port Edelstein et al. 2023's regularized geodesic distances (`rdg_ADMM.m` plus the
mesh operators it needs) to Python, validate against the existing MATLAB
implementation, and eventually retire MATLAB from the research loop.

The current pipeline calls `external/matlab_rgd/demo.m`, which is a `parfor`
loop over `nv` sources invoking `rdg_ADMM` per source and collecting
`argmax(u_i)` into a `max_indices_*.txt` table. Python/Unity downstream reads
that table. Replacing `demo.m` with a Python equivalent is sufficient — we do
not need to port `rdg_allpairs_admm.m` (Lana's pipeline never uses it).

## 1. Analysis

### What `rdg_ADMM.m` actually does (Dirichlet regularizer, `reg='D'`)

Solves, for a single source `x0`, the ADMM problem

    min_{u,z}  sum_f area(f) * ||z_f - G u|_f||^2  + alpha * u^T Ww u,
    s.t.       z is a unit vector field per face (|z_f| <= 1),
               u[x0] = 0.

with

- `alpha = alpha_hat * sqrt(sum(va))`  — area-scaled regularizer
- `rho   = 2 * sqrt(sum(va))`          — ADMM penalty
- over-relaxation factor `alphak = 1.7`
- primal/dual tolerances `ABSTOL = 0.5e-5`, `RELTOL = 1e-2`
- varying penalty: scale `rho` by `+/-2` if `r_norm / s_norm` leaves `[1/10, 10]`

Per iteration (prefactored `L = chol(Ww_p)` once at setup):

  b       = va_p - div(y) + rho * div(z)
  u_p     = L^{-T} L^{-1} P' b  /  (alpha + rho)      # Cholesky solve
  Gx      = G_p * u_p
  z       = (1/rho) y + Gx
  z_f     = z_f / max(1, |z_f|)                       # unit-ball projection per face
  y       = y + rho * (alphak * Gx + (1 - alphak) * z_old - z)
  div_z, div_y = div(z), div(y)
  [update rho based on residual ratio]

`u[x0] = 0` is enforced by eliminating the source row/column from
`Ww`, `G`, `va` (subscript `_p` for "punctured" in the MATLAB source).
`div_p = G_p' * diag(ta, ta, ta)` — area-weighted transpose of the gradient,
NOT the `1/va`-normalized divergence `D = -diag(1/va) G^T diag(ta)` in `MeshClass.DD`.

### Operators required

All computable from vertices + faces with standard formulas:

- `ta` (nf,)   : face areas via cross product of edge vectors.
- `va` (nv,)   : barycentric vertex areas = `(1/3) * sum of incident ta`.
- `Nf` (nf,3)  : unit face normals (orientation convention matters!).
- `Ww` (nv x nv sparse, SPSD): cotangent Laplacian. **Already validated** to
  `max|py-mat| = 4e-10` on the icosphere in `compare_matlab_vs_python.py`.
- `G`  (3nf x nv sparse): gradient operator such that `(G u).reshape(3, nf).T`
  gives the gradient of the piecewise-linear vertex field `u` on each face
  (as a 3-vector in world coordinates). MATLAB formula:

      G[f, i] = -cross(Nf_f, v_j - v_k) / (2 * ta_f)
      G[f, j] = +cross(Nf_f, v_i - v_k) / (2 * ta_f)
      G[f, k] = -cross(Nf_f, v_i - v_j) / (2 * ta_f)

### What we already have

- `validation/validate_rgd_toy.py`       Python reference `cotLaplacian` on
                                         the icosphere, matches MATLAB to
                                         4e-10.
- `validation/run_matlab_validation.m`   MATLAB dumps `cotLaplacian` +
                                         `rdg_ADMM` `u` for 5 `alpha_hat`
                                         values on the icosphere.
- `validation/compare_matlab_vs_python.py`  Diffs MATLAB vs Python references.
                                         Argmax agreement 100%; rel L2 shape
                                         error 0.9% at `alpha_hat = 0.001`;
                                         clean monotone convergence in
                                         `alpha_hat`.

## 2. Correctness protocol

Five tiers, each blocking the next. Stop and debug at the first failure.

### Tier 0 — Operator-level diffs (icosphere, 642 verts)

| Quantity | Tolerance (max abs) | Notes |
|---|---|---|
| `ta` | 1e-12 | deterministic cross-product |
| `va` | 1e-12 | sum of `ta/3` |
| `Nf` | 1e-12 (up to face orientation) | must match MATLAB orientation convention |
| `Ww` | 1e-10 | already done |
| `G`  | 1e-10 on `nnz` | sparse matrix diff |

### Tier 1 — Single-source solver vs MATLAB on icosphere

At each `alpha_hat` in `{0.001, 0.01, 0.05, 0.25, 1.0}` (we already have MATLAB
outputs for these):

- Argmax agreement: **100%** (MATLAB antipode is vertex 4 = Python vertex 3).
- `u[source]` exactly 0.
- `u >= 0` everywhere.
- `max |u_py - u_mat|` < 1e-3 (absolute).
- `||u_py - u_mat|| / ||u_mat||` < 1e-3 (relative L2).

### Tier 2 — Single-source on the real CIELAB neural-bounded mesh

- Generate (once) a MATLAB gold standard: run `rdg_ADMM` at `alpha_hat = 1.25`,
  source = 1, on `external/matlab_rgd/RGB2CIELAB_neural_1.off`. Save `u_gold.csv`.
- Port must match to relative L2 < 1e-3 on the same mesh.
- Plateau argmax mismatches (where `u[#1] - u[#2] < 1e-4`) are acceptable.

### Tier 3 — Downstream LUT (what the paper actually depends on)

Run the full pipeline at `interval = 16` (4913 input colors) with both
backends:

    to_voxels -> [neural bounding done offline] -> smooth_mesh ->
    (MATLAB demo.m | Python port) -> full_pipeline -> render_lut_cube

Accept if:

- Per-voxel RGB diff, 95th percentile: < 3 / 255 (visually imperceptible).
- Gradient metrics (max, avg, std) within 1% of MATLAB.
- Hue histograms visually indistinguishable.

### Tier 4 — Production: full `interval = 1` run

- Validation: compare the port's final `AllCandidateLABvals_*.npy` against
  MATLAB's for the same config within Tier-3 tolerances.
- Performance target: within 2x of MATLAB on CPU, faster with batched solves
  or GPU.

## 3. Execution plan

Each phase is tagged with its blocking validation gate.

### Phase 1 — Operators (gate: Tier 0)

- Extend `run_matlab_validation.m` to dump `ta, va, Nf, G` as `.npz`-readable
  CSVs for the icosphere.
- New `arlabelvis/rgd/mesh_ops.py` with:
    `face_normals_and_areas(V, F) -> (Nf, ta)`
    `vertex_areas(F, ta) -> va`
    `gradient_operator(V, F, Nf, ta) -> csr_matrix of shape (3nf, nv)`
    `cot_laplacian(V, F) -> csr_matrix of shape (nv, nv)`  (promote from
    validation/validate_rgd_toy.py)
- Extend `compare_matlab_vs_python.py` with operator-level diffs.

### Phase 2 — Single-source RGD solver (gate: Tier 1)

- New `arlabelvis/rgd/admm.py` with `rdg_admm(mesh_ops, x0, alpha_hat) -> u`.
- Line-by-line port of `rdg_ADMM.m` for `reg='D'`. Variable names preserved
  where readable. Comments cite the corresponding MATLAB line.
- Cholesky via `scipy.sparse.linalg.factorized` initially; upgrade to
  `scikit-sparse`'s `cholmod` if it matters for Tier 4.
- Validate against the existing `matlab_u_alphahat_*.csv` outputs.

### Phase 3 — Real-mesh validation (gate: Tier 2)

- Run MATLAB on `RGB2CIELAB_neural_1.off` once to get gold `u_D1.csv` and
  a `max_indices.csv`. Save in `validation/gold_cielab_neural/`.
- Run Python port on the same mesh. Diff per Tier 2.
- Expected runtime: MATLAB ~1 minute per source on this mesh (per Kwon et al.
  scaling); we just need one source for Tier 2.

### Phase 4 — All-pairs + pipeline integration (gate: Tier 3)

- New `scripts/run_rgd_python.py` mirroring `demo.m`: loop over sources
  (via `multiprocessing.Pool` initially), collect `argmax(u)` into
  `max_indices_*.txt`. Writes the same filename the existing MATLAB step
  writes, so `scripts/run_rgd.py` / `scripts/full_pipeline.py` work unchanged.
- Add `--backend {matlab,python}` flag.
- Run end-to-end at `interval = 16`, compare final LUTs per Tier 3.

### Phase 5 — Performance (optional; only if Phase 4 is too slow)

- Profile single-source solve. Biggest expected costs:
  - Cholesky back-solves (`~niter * nv`)
  - Sparse matrix multiplies for `G * u_p`, `div_p * z`.
- Batched Cholesky over multiple sources (same factor, different RHS).
- Torch-on-GPU batched CG with a preconditioner.

### Phase 6 — Retire MATLAB from defaults

- Flip pipeline default to `--backend python`.
- MATLAB stays installed as an A/B reference but isn't on the critical path.
- README updated; `external/matlab_rgd/` marked as reference-only.

## 4. Risks + fallback

- **ADMM tolerances**: MATLAB's ABSTOL/RELTOL may need small adjustment for
  Python floating-point. We compare outputs, not iteration counts.
- **Sparse Cholesky speed**: scipy's default solver is slower than MATLAB's.
  Fallback: `scikit-sparse` (bindings to SuiteSparse's cholmod). Final
  fallback: preconditioned CG.
- **Sign/orientation conventions**: Nf, G, div each have sign choices. Only
  the *combinations* that appear in `rdg_ADMM` need to match MATLAB's — e.g.,
  we never need `D = -diag(1/va) G^T diag(ta)` directly, only `G^T diag(ta)`.
- **MATLAB stays available throughout the port.** Phases 1-5 are additive;
  Phase 6 is the only switch.
