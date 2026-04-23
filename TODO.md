# TODO — deferred work

Current Python RGD backend is correct and byte-equivalent to MATLAB but **1.4× slower** on a 1002-vertex neural-bounded CIELAB all-pairs (18 s vs 13 s with 16 workers, BLAS pinned to 1 thread/worker in [arlabelvis/rgd/allpairs.py::_init_worker](arlabelvis/rgd/allpairs.py)). Two real paths to go sub-MATLAB.

## Option A — Numba-JIT the inner ADMM iteration  (~2× more)

Per-source time is ~112 ms on the neural mesh, ~309 ADMM iterations = 0.36 ms/iter. Where it goes:

| | share | notes |
|---|---:|---|
| scipy sparse solve (SuperLU backsolve) | 25 % | 0.11 ms/solve; already near-optimal |
| sparse matvec × 3 per iter (`G_p @ u`, `div_p @ z`, `div_p @ y`) | 25 % | ~30 μs Python dispatch each |
| unit-ball normalization | 11 % | reshape + norm + broadcast |
| numpy array arithmetic (rest) | 39 % | many small ops, Python-dispatch-bound |

**What to JIT**: the inner `for ii in range(niter):` loop body in [arlabelvis/rgd/admm.py::rdg_admm](arlabelvis/rgd/admm.py). Numba can't consume scipy's sparse classes directly; hand-roll CSR matvec as a `@njit` function that takes `(indptr, indices, data, x, out)`. Keep the sparse solve (`factorized(...)`) out of the JIT — call it from Python between JIT-compiled blocks. Most of the saving is collapsing the dozen per-iter numpy calls into one compiled function.

Estimate: 2-3× on the 112 ms per-source work → all-pairs drops from 18 s to 6-9 s, i.e. better than MATLAB's 13 s.

Risks: subtle floating-point differences from Numba's op ordering; revalidate against the full regression in `validation/compare_matlab_vs_python.py`.

## Option B — torch + GPU, batched across sources  (~5-10× more)

The current pool runs per-source ADMM in parallel *processes*. A GPU version would run multiple sources' u, y, z as stacked batch dimensions on one GPU, with `torch.sparse.solve` factoring each source's Ww_p. Reformulation is non-trivial: the per-source boundary condition (eliminate row/col x0) gives each source a different matrix, so batched factorization is awkward.

Cleaner alternative: factor `Ww` (the full nv × nv Laplacian) once with a rank-1 regularization + Schur complement per source, so one global factor serves every source. Either way, the per-iter work is just dense tensor ops — GPU shines.

Estimate: 5-10× over current 18 s for the 1002-vertex case; the win grows with mesh size since GPU amortization kicks in at ~10K verts.

Risks: bigger rewrite, torch's sparse-solve on Windows has its own quirks, and CHOLMOD-equivalent on GPU (`cuDSS`, `MAGMA`) requires a CUDA toolkit install beyond the default torch wheel.

## What did NOT pan out

- `scikit-sparse` (the usual CHOLMOD wrapper): install blocked on Windows — no wheel, and pip can't build from source without SuiteSparse C libs present. Would need conda or vcpkg. But see the next item — CHOLMOD is available via a different wrapper.
- `pypardiso` (Intel MKL PARDISO): installs from pip but throws `OSError: access violation` on our Ww_p. Likely MKL + MSVC interaction on Windows 11.
- Increasing worker count alone: hits a ceiling at 16 workers with 20 physical cores; the real bottleneck is per-source work, not parallel efficiency.

## Integrated — `cholespy` (CHOLMOD via uv)

`cholespy` ships prebuilt Windows wheels with SuiteSparse bundled; installs
cleanly via `uv add cholespy`. Integrated into
[arlabelvis/rgd/admm.py::_build_spd_solver](arlabelvis/rgd/admm.py) with a
scipy-SuperLU fallback. All 8 regression tiers still pass; output matches
scipy's to 8e-15.

**Per-solve** on our Ww_p: 50 μs (cholespy) vs 99 μs (scipy) — **2× faster**.
**Per-source serial** (factor + 300 solves): 26 ms (cholespy) vs 42 ms (scipy) — **1.6× faster**.
**All-pairs at 16 workers**: both land at ~18-22 s — tied within noise.

The parallel wash is because factor + solve together is only ~15% of per-source
wall-clock (sparse matvecs + numpy array ops dominate), so halving that slice
is invisible against the rest.  Worth it for interactive single-source runs
and as groundwork if we ever do Numba or GPU — the SPD solver is no longer a
bottleneck; it's also not Python-SuperLU.

## Other deferred items (lower priority)

- **Run the interval=1 pipeline end-to-end** — [arlabelvis/bounding.py::neural_bounded_mesh](arlabelvis/bounding.py) now produces a clean 495K-vertex watertight neural-bounded mesh at full fidelity (no decimation). All-pairs RGD on 495K sources is ~14 h MATLAB / ~40 h Python. Overnight commitment; drops a thesis-fidelity LUT at the end.
- **Short-paper TODOs** — 7 content gaps flagged inline in [ieeevis2026/paper.tex](../Lana%20Yang-Maccini%20Senior%20Thesis/ieeevis2026/paper.tex) (user study, ΔE₀₀ smoothness citation, α̂-selection justification, generality demo, timing/LUT-size analysis, neural-bounding architecture, single test sequence). Belongs in the paper-editing turn, not the code repo.
- **`curved_hessian` MATLAB MEX build** — the Stein 2020 repo is in `external/stein2020_curved_hessian/`; got past CMake compat and M_PI issues, blocked at linker (`mx.lib` vs `libmx.lib` mismatch in libigl's FindMATLAB for modern MATLAB). Python already has equivalent via `igl.curved_hessian_energy`; this is only needed for byte-for-byte MATLAB cross-validation of `reg='H'`, which Lana's pipeline doesn't use.
