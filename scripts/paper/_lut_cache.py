"""Content-addressable LUT cache for the paper pipeline.

Hashes a LUT config (color space, smoothing, sigma, distance, alpha_hat,
interval, voxel_dim) into a filename under `data/luts/cache/`. Calling
`get_lut(**config)` returns a dense (256,256,256,3) LUT, producing it from
scratch if the cache miss means running the appropriate pipeline stages:

    generate_LABs  ->  mesh construction (sphere / neural / convex hull / none)
                   ->  farthest-color search (Euclidean / DeltaE76 / RGD)
                   ->  interpolate to 256^3

Cached LUTs are stored as `.npy` (float32 LAB values) + a `.meta.json` that
records the config for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "luts" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


Space = Literal["CIELAB", "OKLAB", "sRGB"]
Smoothing = Literal["none", "sphere", "neural", "gaussian", "convex_hull"]
# Scoring function used for the farthest-color argmax. This is the choice of
# *how we rank candidate hull (or mesh) vertices*, not a pure mathematical
# distance — "RGD" is a whole mesh-geodesic algorithm, "DeltaE94"/"DeltaE00"
# are CIELAB-specific perceptual reweightings, and "Euclidean" is the generic
# L2 norm in whatever intermediate space we're in.
#
# ΔE₇₆ is *not* a value here — by definition it equals Euclidean distance in
# CIELAB, so the Kwon baseline is spelled `(metric="Euclidean", space="CIELAB",
# interp_space="CIELAB")` via the named config `LUT_CIELAB_DELTAE76`.
Metric = Literal["Euclidean", "DeltaE94", "DeltaE00", "RGD"]
# Where the LUT interpolates between chosen farthest-color samples:
#   "sRGB"   -> store integer sRGB of the chosen hull point; trilinear blend
#               happens in sRGB.
#   "CIELAB" -> store continuous CIELAB of the chosen hull point; trilinear
#               blend happens in CIELAB, then lab2rgb at render time. This is
#               the ΔE₇₆/₉₄/₀₀ Kwon-style output path.
InterpSpace = Literal["sRGB", "CIELAB"]


# Cross-field validity rules. Python can't express these in the type system,
# so `LutConfig.__post_init__` enforces them at construction time rather than
# waiting for a runtime failure deep inside the farthest-color routine.
#
# Key : (metric, space) or (metric, smoothing) predicate string. Value: why.
_VALID_COMBOS = {
    "DeltaE94_needs_cielab": (
        "`metric='DeltaE94'` uses the CIE94 formula, which is calibrated for "
        "CIELAB coordinates; running it on sRGB/OKLAB points is not meaningful."
    ),
    "DeltaE00_needs_cielab": (
        "`metric='DeltaE00'` uses the CIEDE2000 formula, which is calibrated "
        "for CIELAB coordinates; running it on sRGB/OKLAB points is not meaningful."
    ),
    "RGD_needs_mesh": (
        "`metric='RGD'` needs a mesh; set `smoothing` to one of "
        "'sphere', 'neural', 'gaussian', or 'convex_hull'."
    ),
    "sRGB_no_cielab_interp": (
        "`space='sRGB'` can't interpolate in CIELAB — there's no CIELAB "
        "coordinate to store. Use `interp_space='sRGB'`."
    ),
}


@dataclass(frozen=True)
class LutConfig:
    space: Space = "CIELAB"
    smoothing: Smoothing = "none"
    sigma: float = 0.0            # only used when smoothing == "gaussian"
    metric: Metric = "Euclidean"
    alpha_hat: float = 0.0        # only used when metric == "RGD"
    interval: int = 16            # input-sRGB grid stepSize
    voxel_dim: int = 256          # for neural binvox mesh extraction
    interp_space: InterpSpace = "sRGB"

    # Fields added AFTER the original key schema are omitted from the hash
    # when they hold their default value, so pre-existing cache entries stay
    # valid. When a new field takes a non-default value, it joins the hash
    # and produces a fresh key.
    _POST_SCHEMA_FIELDS = {"interp_space": "sRGB"}
    # Serialise renamed fields under their historical name so cache keys for
    # pre-rename entries keep resolving. Internally everyone reads the new name.
    _HASH_FIELD_RENAMES = {"metric": "distance"}

    def __post_init__(self):
        if self.metric in ("DeltaE94", "DeltaE00") and self.space != "CIELAB":
            key = f"{self.metric}_needs_cielab"
            raise ValueError(f"{self}: {_VALID_COMBOS[key]}")
        if self.metric == "RGD" and self.smoothing == "none":
            raise ValueError(f"{self}: {_VALID_COMBOS['RGD_needs_mesh']}")
        if self.space == "sRGB" and self.interp_space == "CIELAB":
            raise ValueError(f"{self}: {_VALID_COMBOS['sRGB_no_cielab_interp']}")

    def key(self) -> str:
        d = asdict(self)
        for name, default in self._POST_SCHEMA_FIELDS.items():
            if d.get(name) == default:
                d.pop(name, None)
        for new_name, old_name in self._HASH_FIELD_RENAMES.items():
            if new_name in d:
                d[old_name] = d.pop(new_name)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def lut_path(self) -> Path:
        return CACHE_DIR / f"{self.key()}.npy"

    def meta_path(self) -> Path:
        return CACHE_DIR / f"{self.key()}.meta.json"


def _build_mesh(cfg: LutConfig, allLABs, allRGBs):
    """Return (vertices, faces) in whichever space the RGD step expects."""
    if cfg.smoothing == "none":
        return None, None
    if cfg.smoothing == "convex_hull":
        hull = ConvexHull(allLABs)
        V = allLABs[hull.vertices]
        remap = -np.ones(len(allLABs), dtype=np.int64)
        remap[hull.vertices] = np.arange(len(hull.vertices))
        F = remap[hull.simplices]
        return V, F
    if cfg.smoothing == "sphere":
        # Sphere smoothing = bound the gamut by its enclosing sphere, then run
        # RGD on a direct icosphere triangulation of that sphere. Using an
        # icosphere (rather than the hull of sphere-mapped grid points) gives
        # a clean, uniformly-tessellated mesh at any input grid density.
        import trimesh
        from arlabelvis.bounding import bindLABtoSphere
        bound = bindLABtoSphere(allLABs.copy(), allRGBs)
        center = bound.mean(axis=0)
        radius = float(np.linalg.norm(bound - center, axis=1).max())
        ico = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        V = np.asarray(ico.vertices) + center
        F = np.asarray(ico.faces)
        return V, F
    if cfg.smoothing == "neural":
        from arlabelvis.bounding import neural_bounded_mesh
        binvox = ROOT / f"data/neural_bounding_{cfg.space}_{cfg.voxel_dim}.binvox"
        if not binvox.exists():
            raise FileNotFoundError(
                f"neural-bounded binvox missing: {binvox}. Train it first or "
                f"use a different smoothing mode."
            )
        # Keep the mesh modest so all-pairs RGD on it is fast enough for caching.
        mesh = neural_bounded_mesh(str(binvox), cfg.voxel_dim, allLABs,
                                   subdivide=True, target_faces=2000)
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if cfg.smoothing == "gaussian":
        # Voxel-based Gaussian-smoothed gamut mesh. Always build from the full
        # 256^3 LAB point cloud (independent of cfg.interval) so the voxel
        # grid is fully populated — small-sigma Gaussian blurs only connect
        # neighboring voxels, so sparse sampling fragments the mesh.
        # pointsToMesh re-remeshes after decimation, blowing the face count
        # back up; cap it afterwards with a second quadric decimation so
        # all-pairs RGD on the result stays tractable.
        from arlabelvis.meshing import generate_LABs, pointsToMesh
        import open3d as o3d
        _, dense_LABs = generate_LABs(stepSize=1)
        mesh = pointsToMesh(dense_LABs, sigma=cfg.sigma, vox=cfg.voxel_dim,
                            target_faces=2000)
        if len(mesh.faces) > 2000:
            m = o3d.geometry.TriangleMesh()
            m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            m.triangles = o3d.utility.Vector3iVector(mesh.faces)
            m = m.simplify_quadric_decimation(target_number_of_triangles=2000)
            m.remove_unreferenced_vertices()
            V = np.asarray(m.vertices); F = np.asarray(m.triangles)
        else:
            V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        print(f"[lut_cache] gaussian mesh after post-decimation: "
              f"{len(V)} verts, {len(F)} faces")
        return V, F
    raise ValueError(f"unknown smoothing={cfg.smoothing}")


def _compute_furthest(cfg: LutConfig, allLABs, allRGBs, V, F):
    """Return the per-input farthest-color array.

    Output format depends on ``cfg.interp_space``:
      - "sRGB"   -> (N, 3) float in [0, 255], the grid sRGB of the chosen hull
                    vertex. Downstream trilinear interp happens in sRGB.
      - "CIELAB" -> (N, 3) float in CIELAB, the continuous hull LAB coordinate
                    of the chosen hull vertex. Downstream trilinear interp
                    happens in CIELAB; lab2rgb runs at render time. For OKLAB
                    space this stores continuous OKLAB coordinates.
    """
    # ----- Euclidean in whichever LAB-like / sRGB space ---------------------
    if cfg.metric == "Euclidean":
        # sRGB-space "Euclidean": trivial 8-corner map, always sRGB-valued.
        if cfg.space == "sRGB":
            from arlabelvis.distances import furthest_euclidean_rgb
            if cfg.interp_space != "sRGB":
                raise ValueError("sRGB-space Euclidean only supports sRGB interp")
            return furthest_euclidean_rgb(allRGBs)
        return _hull_farthest(allLABs, allRGBs, cfg.interp_space, metric="euclidean")

    # ----- Perceptual ΔE metrics in CIELAB ----------------------------------
    if cfg.metric in ("DeltaE94", "DeltaE00"):
        if cfg.space != "CIELAB":
            raise ValueError(f"{cfg.metric} is only defined in CIELAB, got {cfg.space}")
        return _hull_farthest(allLABs, allRGBs, cfg.interp_space,
                              metric=cfg.metric)

    if cfg.metric == "RGD":
        if V is None:
            raise ValueError("RGD requires a mesh (smoothing != 'none')")
        from arlabelvis.rgd.allpairs import compute_all_pairs_argmax
        # Compute per-mesh-vertex argmax under RGD.
        max_idx = compute_all_pairs_argmax(
            V, F, cfg.alpha_hat, one_indexed=True, progress_every=0
        )
        tmp_path = CACHE_DIR / f"{cfg.key()}_max_idx.csv"
        np.savetxt(tmp_path, max_idx.reshape(1, -1), fmt="%d", delimiter=",")
        return _rgd_furthest(V, allLABs, allRGBs, str(tmp_path), cfg.interp_space)

    raise ValueError(f"unknown metric={cfg.metric}")


def _hull_farthest(allLABs, allRGBs, interp_space: str, metric: str):
    """Shared path for {Euclidean, DeltaE94, DeltaE00} on LAB-like points.

    Builds the convex hull of ``allLABs`` once, then for each input finds the
    hull vertex that maximizes ``metric`` and emits its sRGB or LAB coord.
    """
    lab = np.asarray(allLABs, dtype=np.float32)
    hull = ConvexHull(lab)
    hull_idx = hull.vertices
    hull_lab = lab[hull_idx].astype(np.float32)
    hull_rgb = np.asarray(allRGBs, dtype=np.float32)[hull_idx]

    if metric == "euclidean":
        # Vectorised pairwise squared-distance argmax, chunked to cap memory.
        N = len(lab)
        sel = np.empty(N, dtype=np.int64)
        chunk = 10_000
        for i in range(0, N, chunk):
            block = lab[i:i + chunk]
            d2 = np.sum((block[:, None, :] - hull_lab[None, :, :]) ** 2, axis=-1)
            sel[i:i + chunk] = np.argmax(d2, axis=1)
    elif metric == "DeltaE94":
        from skimage.color import deltaE_ciede94
        N = len(lab); sel = np.empty(N, dtype=np.int64)
        for i in range(N):
            d = deltaE_ciede94(np.tile(lab[i], (len(hull_lab), 1)), hull_lab)
            sel[i] = int(np.argmax(d))
    elif metric == "DeltaE00":
        from skimage.color import deltaE_ciede2000
        N = len(lab); sel = np.empty(N, dtype=np.int64)
        for i in range(N):
            d = deltaE_ciede2000(np.tile(lab[i], (len(hull_lab), 1)), hull_lab)
            sel[i] = int(np.argmax(d))
    else:
        raise ValueError(f"unknown metric {metric!r}")

    if interp_space == "sRGB":
        return hull_rgb[sel]
    if interp_space == "CIELAB":
        return hull_lab[sel]
    raise ValueError(f"unknown interp_space {interp_space!r}")


def _rgd_furthest(V, allLABs, allRGBs, max_idx_path, interp_space: str):
    """Per-input farthest-vertex lookup for RGD. Uses the vertex-to-vertex
    argmax written by the RGD all-pairs solver.

    Output is either the grid sRGB of the closest input point to the chosen
    mesh vertex (sRGB interp) or the chosen mesh vertex's LAB coordinate
    (CIELAB interp)."""
    from scipy.spatial import cKDTree
    allLABs = np.asarray(allLABs, dtype=np.float64)
    allRGBs = np.asarray(allRGBs)
    V = np.asarray(V, dtype=np.float64)

    mesh_tree = cKDTree(V)
    LABtoVertices = mesh_tree.query(allLABs)[1].astype(np.int64)
    max_indices = np.loadtxt(max_idx_path, delimiter=",", dtype=np.int64)

    furthest_vert = np.where(LABtoVertices < len(max_indices),
                             max_indices[LABtoVertices] - 1, 0)
    if interp_space == "CIELAB":
        return V[furthest_vert].astype(np.float32)
    # sRGB interp: map each chosen vertex back to its nearest input LAB, output that LAB's sRGB.
    lab_tree = cKDTree(allLABs)
    vert_to_lab = lab_tree.query(V)[1].astype(np.int64)
    return allRGBs[vert_to_lab[furthest_vert]].astype(np.float32)


def get_lut(cfg: Optional[LutConfig] = None, **kwargs) -> np.ndarray:
    """Produce-or-load a dense 256^3 LUT for the given config.

    Returned array is (256, 256, 256, 3) float32; contents are the farthest-
    color output values (sRGB 0..255 or LAB, depending on cfg.metric).
    """
    if cfg is None:
        cfg = LutConfig(**kwargs)
    else:
        assert not kwargs, "pass cfg or kwargs, not both"

    lut_path = cfg.lut_path()
    if lut_path.exists():
        return np.load(lut_path)

    print(f"[lut_cache] MISS {cfg.key()}  -> building {cfg}")
    t0 = time.perf_counter()

    from arlabelvis.meshing import generate_LABs
    from arlabelvis.interpolate import interpolate_interval
    allRGBs, allCIELABs = generate_LABs(stepSize=cfg.interval)
    # generate_LABs always returns CIELAB; convert to the requested color space
    # so downstream mesh + farthest-color steps operate in that space.
    if cfg.space == "CIELAB":
        allLABs = allCIELABs
    elif cfg.space == "OKLAB":
        from arlabelvis.colors import sRGBtoOKLAB
        allLABs = sRGBtoOKLAB(allRGBs)
    elif cfg.space == "sRGB":
        allLABs = allRGBs.astype(np.float64)
    else:
        raise ValueError(f"unknown space={cfg.space!r}")
    V, F = _build_mesh(cfg, allLABs, allRGBs)
    furthest = _compute_furthest(cfg, allLABs, allRGBs, V, F)

    interpolate_interval(allRGBs, furthest, str(lut_path), cfg.interval)

    cfg.meta_path().write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    elapsed = time.perf_counter() - t0
    print(f"[lut_cache] HIT -> built in {elapsed:.1f}s, saved {lut_path.name}")
    return np.load(lut_path)
