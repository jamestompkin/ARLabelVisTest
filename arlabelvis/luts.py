"""LUT construction pipeline for farthest-color label selection.

Every supported method is one configuration of a five-stage pipeline:

    1. generate_input_grid   -> input grid in cfg.input_space, plus its
                                 sRGB image (used for indexing the dense LUT)
    2. to_working_space      -> working coordinates in cfg.working_space
    3. build_candidates      -> CandidateSet (vertices + optional faces + rgb)
    4. score_argmax          -> per-input candidate index
    5. render_candidate      -> per-input output colour
    then dense_lut_from_sparse produces the dense 256^3 sRGB-indexed LUT
    via *nearest-neighbour* assignment from each seed (see
    ``cfg.interp_method``). Trilinear blending is available behind
    ``interp_method='linear'`` but is not the default: blending two
    farthest-point picks yields sRGB triplets that aren't farthest from
    anything, which silently breaks any palette / K_eff / hue-diversity
    analysis downstream.

Top-level API:

- ``LutConfig`` — the configuration dataclass (what to build).
- ``LutCache`` — on-disk content-addressable store (where to cache).
- ``build_lut(cfg)`` — run the pipeline, return the array. No I/O side-effects.
- ``get_lut(cfg, cache=DEFAULT_CACHE)`` — load-or-build-and-cache.
- ``lut_to_srgb_u8(lut, cfg_or_interp_space)`` — display-ready ``uint8``.

The stage functions (``to_working_space``, ``build_candidates``,
``score_argmax``, ``render_candidate``) take the specific fields they need,
not a full ``LutConfig``, so they're reusable from other contexts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as _mp
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

_log = logging.getLogger(__name__)


def _default_data_root() -> Path:
    """Where on-disk defaults live. If the library is inside a developer
    checkout, use the repo's ``data/`` directory (preserves existing caches,
    binvoxes, etc.). Otherwise — installed to site-packages, or a packaged
    bundle — fall back to ``~/.arlabelvis/`` so we never try to write inside
    the package install.

    A checkout is detected by the presence of ``scripts/paper/_configs.py``
    as a sibling of the ``arlabelvis/`` package — that layout can only come
    from an in-place developer clone.
    """
    here = Path(__file__).resolve().parents[1]
    if (here / "scripts" / "paper" / "_configs.py").exists():
        return here / "data"
    return Path.home() / ".arlabelvis"


_DATA_ROOT = Path(os.environ.get("ARLABELVIS_HOME", str(_default_data_root())))


# ---------------------------------------------------------------------------
# Configuration schema.
# ---------------------------------------------------------------------------
# One colour-space alphabet for every axis. Input / working / output can each
# independently be any of these.
Space = Literal["CIELAB", "OKLAB", "sRGB"]
# The base geometry of the candidate set:
#   "hull"   - exact convex hull of the working-space points
#   "sphere" - circumscribing icosphere (outer bound)
#   "neural" - in-process ReLU MLP trained here on the working-space gamut;
#              level set extracted exactly via polytope-arrangement clipping
#              (no binvox round-trip). ``shape_neural_bound_bias`` selects
#              inner / balanced / outer character; see ``neural_bounding`` for
#              the design.
Shape = Literal["hull", "sphere", "neural"]
# Optional operation applied to the base geometry to produce the final
# candidate mesh. "gaussian" = voxelise + Gaussian blur (sigma) + marching
# cubes. Currently only implemented for shape='hull'; see __post_init__.
Smoothing = Literal["none", "gaussian"]
# The scorer — how we rank candidate vertices. "RGD" is a whole mesh-geodesic
# algorithm; "DeltaE94"/"DeltaE00" are CIELAB-specific perceptual reweightings;
# "Euclidean" is L2 in whatever space the coordinates are in. ΔE₇₆ is not a
# value here — by definition it equals Euclidean in CIELAB.
Metric = Literal["Euclidean", "DeltaE94", "DeltaE00", "RGD"]
# Output and input spaces reuse ``Space`` — each axis can independently be
# CIELAB / OKLAB / sRGB. The output path blends in that coordinate system
# and converts back to display sRGB at render time.


@dataclass(frozen=True)
class LutConfig:
    """LUT build configuration. Fields are grouped by pipeline axis:

    - *Space axis*: ``input_space``, ``working_space``, ``output_space``.
      Each is independently CIELAB / OKLAB / sRGB.
    - *Shape axis* (stage 3, ``build_candidates``): ``shape``, ``smoothing``,
      ``shape_gaussian_sigma`` (parameter of the gaussian smoothing operation).
    - *Metric axis* (stage 4, ``score_argmax``): ``metric``,
      ``metric_rgd_alpha_hat`` (parameter of the RGD metric's regularisation).
    - *Resolution knobs*: ``interval`` (sRGB-grid stride for seed sampling)
      and ``voxel_dim`` (only used by the gaussian smoothing path). The
      dense LUT is always 256^3 sRGB-indexed regardless.

    The axis-prefixed parameter names make explicit which operation they
    parameterise. Internally, the pipeline stages use the short canonical
    names (``sigma``, ``alpha_hat``) that match the RGD / Gaussian literature.
    """
    # Space axis
    input_space: Space = "sRGB"         # 256^3 sampled grid's coordinate system
    working_space: Space = "CIELAB"     # where stages 2-5 operate
    output_space: Space = "sRGB"        # storage for the per-voxel LUT value
    # Shape axis
    shape: Shape = "hull"
    smoothing: Smoothing = "none"
    shape_gaussian_sigma: float = 0.0   # used only when smoothing == "gaussian"
    # Neural-shape hyperparameters (used only when shape == "neural").
    # ``shape_neural_bound_bias`` is the asymmetric-BCE class-weight ratio
    # that the in-process MLP trains under: < 0 → inner bound, 0 → balanced,
    # > 0 → outer bound. ``|bias|=3`` ≈ Liu 2024's converged ratio.
    shape_neural_bound_bias: float = -1.0
    shape_neural_iterations: int = 20_000
    # Metric axis
    metric: Metric = "Euclidean"
    metric_rgd_alpha_hat: float = 0.0   # used only when metric == "RGD"
    # Resolution knobs
    interval: int = 16
    # ``voxel_dim`` is *only* read by ``_hull_gaussian_candidates`` (the only
    # voxelising path); kept named broadly because the field is part of the
    # cache key on every existing build. Has no effect when
    # ``smoothing != 'gaussian'``.
    voxel_dim: int = 256
    # Dense-LUT bake-out method. ``'nearest'`` (the default) preserves the
    # algorithmic farthest-point partition exactly: every dense voxel
    # inherits whichever seed's pick is closest in sRGB, so the dense LUT
    # contains only the K unique algorithmic picks. ``'linear'`` blends
    # adjacent picks trilinearly — yields shader-smooth transitions but
    # produces sRGB triplets that aren't algorithmic picks at all, which
    # silently breaks any palette-diversity / K_eff analysis. Defaulted to
    # nearest after 2026-04-25; pre-fix caches built under linear are
    # numerically wrong for analysis.
    interp_method: Literal["nearest", "linear"] = "nearest"

    def __post_init__(self):
        if self.metric in ("DeltaE94", "DeltaE00") and self.working_space != "CIELAB":
            raise ValueError(
                f"{self}: metric={self.metric!r} uses a CIELAB-calibrated "
                f"formula; running it on {self.working_space} points is not meaningful."
            )
        if self.smoothing == "gaussian" and self.shape != "hull":
            raise ValueError(
                f"{self}: smoothing='gaussian' currently only supports "
                f"shape='hull' (the voxelisation path is fused with the "
                f"gamut point cloud). Extend build_candidates to voxelise "
                f"the sphere/neural base mesh if you need this combination."
            )
        if self.input_space not in ("sRGB", "CIELAB", "OKLAB"):
            raise ValueError(
                f"{self}: unknown input_space={self.input_space!r}"
            )

    def key(self) -> str:
        """SHA-1 of the config for use as a cache key."""
        return hashlib.sha1(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateSet:
    """Finite set of candidate colours the farthest-color scorer picks from.

    ``vertices`` is ``(M, 3)`` in the caller's working space. ``faces`` is
    ``(nf, 3)`` face indices into ``vertices``; every shape produces a
    triangulation so RGD-type scorers can consume it (non-RGD scorers just
    ignore ``faces``). ``rgbs`` is ``(M, 3)`` integer sRGB of the original
    input voxel each candidate came from, used by the sRGB-interp output path.
    """
    vertices: np.ndarray
    faces: np.ndarray
    rgbs: np.ndarray


# ---------------------------------------------------------------------------
# Cache.
# ---------------------------------------------------------------------------
class LutCache:
    """Content-addressable on-disk LUT store.

    Given a ``LutConfig``, yields a canonical ``.npy`` path (``path_for``),
    checks existence (``has``), loads (``load``), and stores (``store``).
    Configs carry no path knowledge of their own. No directory is created
    until the first ``store`` call.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for(self, cfg: "LutConfig") -> Path:
        return self.root / f"{cfg.key()}.npy"

    def meta_path(self, cfg: "LutConfig") -> Path:
        return self.root / f"{cfg.key()}.meta.json"

    def has(self, cfg: "LutConfig") -> bool:
        return self.path_for(cfg).exists()

    def load(self, cfg: "LutConfig") -> np.ndarray:
        return np.load(self.path_for(cfg))

    def load_u8(self, cfg: "LutConfig") -> np.ndarray:
        """Load and immediately convert to display-ready ``uint8`` sRGB.

        Bundles the ``lut_to_srgb_u8(lut, cfg.output_space)`` step so callers
        can't accidentally pass the wrong ``output_space``.
        """
        return lut_to_srgb_u8(self.load(cfg), cfg.output_space)

    def store(self, cfg: "LutConfig", lut: np.ndarray) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(cfg)
        np.save(path, lut)
        self.meta_path(cfg).write_text(
            json.dumps(asdict(cfg), indent=2, sort_keys=True)
        )
        return path


_DEFAULT_LUT_CACHE_ROOT = _DATA_ROOT / "luts" / "cache"
DEFAULT_CACHE = LutCache(root=Path(os.environ.get(
    "ARLABELVIS_LUT_CACHE", str(_DEFAULT_LUT_CACHE_ROOT),
)))


# ---------------------------------------------------------------------------
# Pipeline stages (each takes the specific fields it needs, not a full cfg).
# ---------------------------------------------------------------------------
def to_working_space(working_space: Space, input_space: Space,
                     input_pts: np.ndarray) -> np.ndarray:
    """Convert ``input_pts`` (in ``input_space``) to ``working_space``
    coordinates for farthest-color selection.

    Fast path when ``input_space == working_space``: identity. Otherwise
    pivots through sRGB via ``arlabelvis.colors.convert_color``.
    """
    if input_space == working_space:
        return np.asarray(input_pts, dtype=np.float64)
    from arlabelvis.colors import convert_color
    return convert_color(input_pts, input_space, working_space)


def _nearest_rgb(vertices: np.ndarray, working_pts: np.ndarray,
                 all_rgbs: np.ndarray) -> np.ndarray:
    """For each candidate vertex, return the sRGB of its nearest input voxel."""
    idx = cKDTree(working_pts).query(vertices)[1]
    return all_rgbs[idx]


def _decimate_mesh(vertices: np.ndarray, faces: np.ndarray, *,
                   target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate (V, F) to ``target_faces``. Requires open3d."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(vertices)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def build_candidates(working_pts: np.ndarray, all_rgbs: np.ndarray, *,
                     shape: Shape, smoothing: Smoothing,
                     working_space: Space,
                     input_space: Space = "sRGB",
                     sigma: float = 0.0,
                     voxel_dim: int = 256,
                     neural_bound_bias: float = -1.0,
                     neural_iterations: int = 20_000) -> CandidateSet:
    """Construct the candidate set in ``working_space`` for farthest-color
    selection.

    Two independent axes:

    - ``shape`` selects the *base geometry* of the candidate set:
        * ``'hull'``  : exact convex hull of ``working_pts``.
        * ``'sphere'``: icosphere circumscribing the gamut.
        * ``'neural'``: in-process neurally-bounded gamut — small ReLU MLP
                        trained here, level set extracted exactly from the
                        polytope arrangement. Tunable via
                        ``neural_bound_bias`` (inner ↔ outer bound).
    - ``smoothing`` selects an optional *operation* applied to the base:
        * ``'none'``    : return the base mesh directly.
        * ``'gaussian'``: voxelise + Gaussian blur (``sigma``) + marching
                          cubes. Currently only implemented for
                          ``shape='hull'``; LutConfig validation forbids
                          the other combinations.

    Every shape produces triangulated faces so RGD scorers always have a mesh.
    """
    if smoothing == "gaussian":
        if shape != "hull":
            raise ValueError(
                f"smoothing='gaussian' not implemented for shape={shape!r}; "
                f"only shape='hull' is currently supported."
            )
        return _hull_gaussian_candidates(working_pts, all_rgbs,
                                         input_space=input_space,
                                         working_space=working_space,
                                         sigma=sigma, voxel_dim=voxel_dim)

    if shape == "hull":
        return _hull_candidates(working_pts, all_rgbs)
    if shape == "sphere":
        return _sphere_candidates(working_pts, all_rgbs)
    if shape == "neural":
        return _neural_candidates(working_pts, all_rgbs,
                                  bound_bias=neural_bound_bias,
                                  iterations=neural_iterations)
    raise ValueError(f"unknown shape={shape!r}")


def _hull_candidates(working_pts: np.ndarray, all_rgbs: np.ndarray) -> CandidateSet:
    """Exact convex hull of ``working_pts``, with hull simplices as faces."""
    hull = ConvexHull(working_pts)
    vertices = working_pts[hull.vertices]
    rgbs = all_rgbs[hull.vertices]
    remap = -np.ones(len(working_pts), dtype=np.int64)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    faces = remap[hull.simplices]
    # QHull on very dense inputs (e.g., interval=1 OKLAB, ~117k verts)
    # produces sliver triangles that blow up cotangent weights and break
    # RGD Cholesky factorisation. Decimate if the hull is too large.
    HULL_FACE_LIMIT = 10_000
    if len(faces) > HULL_FACE_LIMIT:
        vertices, faces = _decimate_mesh(vertices, faces, target_faces=2000)
        rgbs = _nearest_rgb(vertices, working_pts, all_rgbs)
        _log.info("hull decimated: %d verts, %d faces (was %d verts)",
                  len(vertices), len(faces), len(hull.vertices))
    return CandidateSet(vertices=vertices, faces=faces, rgbs=rgbs)


def _hull_gaussian_candidates(working_pts: np.ndarray, all_rgbs: np.ndarray, *,
                              input_space: Space, working_space: Space,
                              sigma: float, voxel_dim: int) -> CandidateSet:
    """Gaussian-smoothed gamut: voxelise the full dense gamut, blur with
    ``sigma``, marching-cubes the result, decimate. Equivalent to voxelising
    the convex hull for the sRGB-in-CIELAB case (the gamut is convex), and
    that's the regime the paper uses.

    Re-samples the gamut at ``interval=1`` in ``input_space`` and converts
    each dense input point into ``working_space`` so the smoothing operates
    in the same coordinate frame as the rest of the pipeline.
    """
    from arlabelvis.meshing import generate_input_grid, points_to_mesh
    dense_input, _ = generate_input_grid(input_space, interval=1)
    dense_working = to_working_space(working_space, input_space, dense_input)
    mesh = points_to_mesh(dense_working, sigma=sigma, vox=voxel_dim,
                          target_faces=2000)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > 2000:
        vertices, faces = _decimate_mesh(vertices, faces, target_faces=2000)
    _log.info("hull + gaussian: %d verts, %d faces (sigma=%g)",
              len(vertices), len(faces), sigma)
    return CandidateSet(vertices=vertices, faces=faces,
                        rgbs=_nearest_rgb(vertices, working_pts, all_rgbs))


def _sphere_candidates(working_pts: np.ndarray, all_rgbs: np.ndarray) -> CandidateSet:
    """Icosphere circumscribing the gamut (outer bound)."""
    import trimesh
    from arlabelvis.gamut import bind_lab_to_sphere
    bound = bind_lab_to_sphere(working_pts.copy(), all_rgbs)
    center = bound.mean(axis=0)
    radius = float(np.linalg.norm(bound - center, axis=1).max())
    ico = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    vertices = np.asarray(ico.vertices) + center
    faces = np.asarray(ico.faces)
    return CandidateSet(vertices=vertices, faces=faces,
                        rgbs=_nearest_rgb(vertices, working_pts, all_rgbs))


def _neural_candidates(working_pts: np.ndarray, all_rgbs: np.ndarray, *,
                       bound_bias: float, iterations: int) -> CandidateSet:
    """Train a small ReLU MLP on the gamut, then extract its level set
    *exactly* via polytope-arrangement clipping. ``bound_bias`` controls
    inner (negative) vs outer (positive) bound character; see
    ``arlabelvis.neural_bounding`` for the full design."""
    from arlabelvis.neural_bounding import (
        NeuralBoundingParams, neural_bounded_mesh_inprocess,
    )
    params = NeuralBoundingParams(
        bound_bias=bound_bias,
        iterations=iterations,
    )
    mesh = neural_bounded_mesh_inprocess(working_pts.astype(np.float32), params)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(faces) > 2000:
        vertices, faces = _decimate_mesh(vertices, faces, target_faces=2000)
    _log.info("neural mesh: %d verts, %d faces (bound_bias=%+.2f)",
              len(vertices), len(faces), bound_bias)
    return CandidateSet(vertices=vertices, faces=faces,
                        rgbs=_nearest_rgb(vertices, working_pts, all_rgbs))


def score_argmax(working_pts: np.ndarray, candidates: CandidateSet, *,
                 metric: Metric, alpha_hat: float = 0.0) -> np.ndarray:
    """For each input in ``working_pts``, return the index in
    ``candidates.vertices`` that maximises ``metric``.

    Contract: ``(N_inputs, 3) + (M_candidates, 3) -> (N_inputs,) int64``.
    """
    if metric == "Euclidean":
        return _argmax_euclidean(working_pts, candidates.vertices)
    if metric in ("DeltaE94", "DeltaE00"):
        return _hull_delta_e_parallel(working_pts, candidates.vertices, metric)
    if metric == "RGD":
        from arlabelvis.rgd.allpairs import compute_all_pairs_argmax
        per_vert = compute_all_pairs_argmax(
            candidates.vertices, candidates.faces, alpha_hat,
            one_indexed=False, progress_every=0,
        ).astype(np.int64)
        input_to_vert = cKDTree(candidates.vertices).query(np.asarray(working_pts))[1]
        return per_vert[input_to_vert].astype(np.int64)
    raise ValueError(f"unknown metric={metric!r}")


def render_candidate(candidates: CandidateSet, idx: np.ndarray, *,
                     output_space: Space, working_space: Space) -> np.ndarray:
    """Map per-input argmax indices to per-input output values.

    ``output_space='sRGB'``: take the chosen candidate's associated sRGB
    triplet (already attached to the ``CandidateSet``).
    Other ``output_space``: take the chosen candidate's working-space
    coordinate, then convert from ``working_space`` to ``output_space`` if
    the two differ. This guards against the latent bug where
    ``working_space='OKLAB', output_space='CIELAB'`` would have silently
    written OKLAB coordinates into a "CIELAB" LUT.
    """
    if output_space == "sRGB":
        return candidates.rgbs[idx].astype(np.float32)
    verts = candidates.vertices[idx].astype(np.float32)
    if output_space == working_space:
        return verts
    from arlabelvis.colors import convert_color
    return convert_color(verts, working_space, output_space).astype(np.float32)


def _argmax_euclidean(inputs: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Vectorised chunked squared-distance argmax."""
    N = len(inputs)
    sel = np.empty(N, dtype=np.int64)
    chunk = 10_000
    for i in range(0, N, chunk):
        block = inputs[i:i + chunk]
        d2 = np.sum((block[:, None, :] - vertices[None, :, :]) ** 2, axis=-1)
        sel[i:i + chunk] = np.argmax(d2, axis=1)
    return sel


# ---------------------------------------------------------------------------
# Parallel ΔE94/ΔE00 argmax.
# Module-level workers so the spawn context can pickle them on Windows.
# ---------------------------------------------------------------------------
_DE_CACHE: dict = {}


def _de_worker_init(hull_lab_bytes: bytes, hull_shape: tuple, metric: str):
    from arlabelvis.distances import delta_e00, delta_e94
    hull_lab = np.frombuffer(hull_lab_bytes, dtype=np.float32).reshape(hull_shape)
    _DE_CACHE["hull"] = hull_lab
    _DE_CACHE["fn"] = delta_e94 if metric == "DeltaE94" else delta_e00


def _de_worker_chunk(args):
    chunk_bytes, chunk_shape = args
    block = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(chunk_shape)
    d = _DE_CACHE["fn"](block[:, None, :], _DE_CACHE["hull"][None, :, :])
    return d.argmax(axis=1).astype(np.int64)


def _hull_delta_e_parallel(lab: np.ndarray, hull_lab: np.ndarray,
                           metric: str) -> np.ndarray:
    """Per-input argmax hull-vertex under ΔE94/ΔE00, parallel over inputs.

    Chunk size + worker count are tuned to keep ΔE00's ~15 (M,K) float32
    intermediates inside L3; workers cap at 10 (memory-bus saturation past
    that on a 20-logical-core machine).
    """
    N = len(lab); K = len(hull_lab)
    lab = np.ascontiguousarray(lab, dtype=np.float32)
    hull_lab = np.ascontiguousarray(hull_lab, dtype=np.float32)
    hull_bytes = hull_lab.tobytes()
    hull_shape = hull_lab.shape

    chunk = max(64, int(128e6 / (K * 4 * 15)))
    n_workers = min(10, max(1, (_mp.cpu_count() or 2) - 1))

    chunks = []
    for i in range(0, N, chunk):
        block = lab[i:i + chunk]
        chunks.append((block.tobytes(), block.shape))
    n_chunks = len(chunks)
    _log.info("[%s] dispatching %d chunks (chunk=%d, K=%d) to %d workers...",
              metric, n_chunks, chunk, K, n_workers)

    env_pin = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    prev = {k: os.environ.get(k) for k in env_pin}
    os.environ.update(env_pin)

    t0 = time.perf_counter()
    results = np.empty(N, dtype=np.int64)
    try:
        ctx = _mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_de_worker_init,
                      initargs=(hull_bytes, hull_shape, metric)) as pool:
            for idx, argmax in enumerate(pool.imap(_de_worker_chunk, chunks,
                                                    chunksize=1)):
                lo = idx * chunk
                hi = min(N, lo + chunk)
                results[lo:hi] = argmax
                if (idx + 1) % 20 == 0 or idx + 1 == n_chunks:
                    done = hi
                    dt = time.perf_counter() - t0
                    rate = done / dt if dt > 0 else 0
                    eta = (N - done) / rate if rate > 0 else float("inf")
                    _log.info("    [%s] %s/%s (%.1f%%), %s inputs/s, ETA %.1f min",
                              metric, f"{done:,}", f"{N:,}", 100 * done / N,
                              f"{rate:,.0f}", eta / 60)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _log.info("[%s] done in %.1f min", metric, (time.perf_counter() - t0) / 60)
    return results


# ---------------------------------------------------------------------------
# Top-level driver.
# ---------------------------------------------------------------------------
def build_lut(cfg: LutConfig) -> np.ndarray:
    """Run the full pipeline and return the dense ``(256, 256, 256, 3)`` LUT.

    Takes the config at face value — no environment-variable overrides,
    no cache writes. Callers that want env-driven behaviour should go
    through ``get_lut`` (which applies overrides and caches).

    The dense LUT is always sRGB-indexed (256^3 voxels keyed by sRGB byte
    triplets) so downstream rendering can stay simple. ``cfg.input_space``
    only changes *where* the candidate-seed grid is sampled — non-sRGB
    grids land at scattered sRGB coordinates and the dense fill switches
    to nearest-neighbour interpolation.
    """
    from arlabelvis.interpolate import dense_lut_from_sparse
    from arlabelvis.meshing import generate_input_grid

    input_pts, srgb_pts = generate_input_grid(cfg.input_space, interval=cfg.interval)
    # The dense LUT is sRGB-indexed so seeds need integer sRGB triplets for
    # ``_nearest_rgb`` and ``_assemble_dense``. For ``input_space='sRGB'``
    # this is exact; for CIELAB/OKLAB inputs the round+clip discards
    # sub-byte float precision in the seed sRGBs (~0.5/255 worst-case),
    # which matters only for the rare config that wants
    # output_space='sRGB' with a non-sRGB input grid.
    all_rgbs = np.clip(np.round(srgb_pts), 0, 255).astype(np.int64)
    working_pts = to_working_space(cfg.working_space, cfg.input_space, input_pts)
    candidates = build_candidates(
        working_pts, all_rgbs,
        shape=cfg.shape, smoothing=cfg.smoothing,
        input_space=cfg.input_space, working_space=cfg.working_space,
        sigma=cfg.shape_gaussian_sigma, voxel_dim=cfg.voxel_dim,
        neural_bound_bias=cfg.shape_neural_bound_bias,
        neural_iterations=cfg.shape_neural_iterations,
    )
    idx = score_argmax(
        working_pts, candidates,
        metric=cfg.metric, alpha_hat=cfg.metric_rgd_alpha_hat,
    )
    furthest = render_candidate(candidates, idx,
                                  output_space=cfg.output_space,
                                  working_space=cfg.working_space)
    if cfg.input_space == "sRGB":
        return dense_lut_from_sparse(all_rgbs, furthest, interval=cfg.interval,
                                      regular_grid=True, method=cfg.interp_method)
    if cfg.interp_method != "nearest":
        raise NotImplementedError(
            f"interp_method={cfg.interp_method!r} is only supported for "
            f"input_space='sRGB'; non-sRGB input grids only have a "
            f"working-space NN bake-out path."
        )
    # Scattered: NN-fill the dense sRGB-indexed LUT in *working space* so the
    # interpolation respects the perceptual frame the seeds were chosen in.
    # Build a 256^3 sRGB query grid, convert it to working space, then NN
    # against the seeds' working-space coords.
    return _scattered_dense_lut(working_pts, furthest,
                                input_space=cfg.input_space,
                                working_space=cfg.working_space)


def _scattered_dense_lut(seed_working_pts: np.ndarray,
                          seed_values: np.ndarray, *,
                          input_space: Space, working_space: Space) -> np.ndarray:
    """Dense 256^3 sRGB-indexed LUT via NN in working space.

    For each of the 16.7M sRGB voxels, convert it into ``working_space`` and
    pick the seed whose working-space coord is closest. NN-in-working-space
    (rather than NN-in-sRGB) is what makes a non-sRGB ``input_space``
    actually deliver perceptually-aligned interpolation; otherwise the LUT
    inherits sRGB's perceptual non-uniformity at the fill stage and erases
    the benefit of the non-sRGB grid.
    """
    from arlabelvis.colors import convert_color
    r, g, b = np.meshgrid(np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32),
                          np.arange(256, dtype=np.float32), indexing="ij")
    srgb_targets = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    if working_space == "sRGB":
        targets_working = srgb_targets.astype(np.float64)
    else:
        targets_working = convert_color(srgb_targets, "sRGB", working_space)
    tree = cKDTree(np.asarray(seed_working_pts, dtype=np.float32))
    _, idx = tree.query(targets_working.astype(np.float32))
    return np.asarray(seed_values, dtype=np.float32)[idx].reshape(256, 256, 256, 3)


def get_lut(cfg: LutConfig, *, cache: LutCache = DEFAULT_CACHE,
            interval: int | None = None) -> np.ndarray:
    """Produce-or-load a dense 256^3 LUT for the given config.

    Cache hit: ``cache.load(cfg)``.
    Cache miss: ``build_lut(cfg)`` then ``cache.store(cfg, lut)``.

    The optional ``interval`` argument overrides ``cfg.interval`` for this
    call only — useful when iterating quickly in a notebook without
    rewriting the config registry. The override is applied before the
    cache lookup so the smaller-interval rebuild gets its own cache key.
    """
    if interval is not None and interval != cfg.interval:
        cfg = replace(cfg, interval=interval)
    if cache.has(cfg):
        return cache.load(cfg)

    _log.info("MISS %s  -> building %s", cfg.key(), cfg)
    t0 = time.perf_counter()
    lut = build_lut(cfg)
    path = cache.store(cfg, lut)
    _log.info("built in %.1fs, saved %s", time.perf_counter() - t0, path.name)
    return lut


# ---------------------------------------------------------------------------
# Display-ready conversion.
# ---------------------------------------------------------------------------
def lut_to_srgb_u8(lut: np.ndarray,
                   output_space: "Space | LutConfig") -> np.ndarray:
    """Render-ready ``uint8`` sRGB version of a cached LUT.

    Dispatches on ``output_space``:

    - ``'sRGB'``   : LUT already stores 0-255 sRGB floats — clip + cast.
    - ``'CIELAB'`` : LUT stores continuous CIELAB — invert via the in-house
                     :func:`arlabelvis.colors.lab_to_srgb` (matches skimage
                     to ~0.01 byte units after the 2026 matrix fix).
    - ``'OKLAB'``  : LUT stores continuous OKLAB — invert via
                     :func:`arlabelvis.colors.oklab_to_srgb`.

    Both LAB branches use our own conversions to keep the forward / inverse
    paths symmetric (``srgb_to_lab`` ↔ ``lab_to_srgb`` and
    ``srgb_to_oklab`` ↔ ``oklab_to_srgb``).

    Accepts either a ``Space`` literal or a ``LutConfig`` — passing the
    config keeps the LUT paired with the config that produced it so callers
    can't accidentally interpret an OKLAB-interp LUT as sRGB.
    """
    if isinstance(output_space, LutConfig):
        output_space = output_space.output_space
    if output_space == "sRGB":
        return np.clip(lut, 0, 255).astype(np.uint8).reshape(256, 256, 256, 3)
    flat = lut.reshape(-1, 3).astype(np.float32)
    if output_space == "CIELAB":
        from arlabelvis.colors import lab_to_srgb
        rgb = lab_to_srgb(flat)
    elif output_space == "OKLAB":
        from arlabelvis.colors import oklab_to_srgb
        rgb = oklab_to_srgb(flat)
    else:
        raise ValueError(f"unknown output_space={output_space!r}")
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8).reshape(256, 256, 256, 3)
