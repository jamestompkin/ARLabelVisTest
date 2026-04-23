"""All-pairs regularized geodesic distances.

Mirrors MATLAB `demo.m`: for each source vertex `i` in the mesh, solve
`rdg_admm(mesh, i, alpha_hat)` and record `argmax(u)`. Output is an
`nv`-length array of 1-indexed argmaxes so it can be written directly in the
`max_indices_*.txt` format consumed by `arlabelvis.distances.furthest_rgd`.

Parallelized over sources with `multiprocessing.Pool`, mirroring MATLAB's
`parfor`. Each worker builds its own `MeshOps` from (V, F) — sparse operators
don't pickle cheaply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import multiprocessing as mp
import os
import time
import numpy as np

from arlabelvis.rgd.mesh_ops import build_mesh_ops
from arlabelvis.rgd.admm import rdg_admm
from arlabelvis.off import read_off  # re-exported for convenience


# Module-level handles populated by the pool initializer. Cheaper than repickling
# MeshOps (sparse matrices) on every task.
_OPS = None
_ALPHA_HAT = None


def _init_worker(V: np.ndarray, F: np.ndarray, alpha_hat: float) -> None:
    # Per-worker: pin every threadpool we know about to 1 thread so the
    # worker processes don't oversubscribe cores. Without this, each
    # worker's numpy/scipy/cholespy/torch calls spawn N compute threads
    # (N = physical cores); with P workers that's P*N threads contending
    # on the LLC, and parallel efficiency collapses from ~90% to ~20%.
    #
    # MUST happen before any numpy/scipy/torch import inside the worker
    # picks up the threadpool config.
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "TBB_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    # torch has its own threadpools that don't respect OMP_NUM_THREADS.
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass
    global _OPS, _ALPHA_HAT
    _OPS = build_mesh_ops(V, F)
    _ALPHA_HAT = alpha_hat


def _solve_source(source_idx: int) -> int:
    u, _ = rdg_admm(_OPS, x0=source_idx, alpha_hat=_ALPHA_HAT)
    return int(np.argmax(u))


def compute_all_pairs_argmax(
    V: np.ndarray,
    F: np.ndarray,
    alpha_hat: float,
    *,
    n_workers: Optional[int] = None,
    sources: Optional[np.ndarray] = None,
    one_indexed: bool = True,
    progress_every: int = 100,
) -> np.ndarray:
    """Run rdg_admm from every source in `sources` (default: all vertices)
    and return an array of argmaxes.

    Args:
        V, F:       mesh data (0-indexed faces).
        alpha_hat:  regularizer weight.
        n_workers:  multiprocessing pool size. Default: os.cpu_count() - 1, min 1.
        sources:    subset of 0-indexed source vertex indices; default all.
        one_indexed: add +1 to argmaxes for MATLAB compatibility.
        progress_every: print progress every N sources.

    Returns:
        array of shape (len(sources),) of argmaxes.
    """
    nv = V.shape[0]
    if sources is None:
        sources = np.arange(nv, dtype=np.int64)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    print(f"all-pairs RGD: nv={nv}, nf={F.shape[0]}, sources={len(sources)}, "
          f"alpha_hat={alpha_hat}, workers={n_workers}")

    t0 = time.perf_counter()
    results = np.empty(len(sources), dtype=np.int64)

    # For tiny problems, parallel overhead > benefit. Run inline.
    if n_workers <= 1 or len(sources) < 50:
        _init_worker(V, F, alpha_hat)
        for i, src in enumerate(sources):
            results[i] = _solve_source(int(src))
            if progress_every and (i + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(sources) - i - 1) / rate
                print(f"  [{i+1}/{len(sources)}] {rate:.1f} src/s, ETA {eta:.0f}s")
    else:
        with mp.Pool(processes=n_workers,
                     initializer=_init_worker,
                     initargs=(V, F, alpha_hat)) as pool:
            for i, argmax in enumerate(pool.imap(_solve_source, sources.tolist(),
                                                 chunksize=max(1, len(sources) // (n_workers * 8)))):
                results[i] = argmax
                if progress_every and (i + 1) % progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(sources) - i - 1) / rate
                    print(f"  [{i+1}/{len(sources)}] {rate:.1f} src/s, ETA {eta:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"all-pairs done in {elapsed:.1f}s ({len(sources)/elapsed:.1f} src/s)")

    if one_indexed:
        results = results + 1
    return results


# (The OFF reader was previously duplicated here. Re-exported from arlabelvis.off.
#  See module-level `from arlabelvis.off import read_off`.)
