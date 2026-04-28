"""In-process neural gamut bounding.

A small ReLU MLP (matching Liu 2024 / external `neural_bounding`
submodule's 3→50→50→1 architecture) is trained to approximate the
indicator function of the working-space gamut. The trained network's
level set ``{x : f(x) >= threshold}`` is a piecewise-linear surface — a
ReLU MLP partitions input space into convex polytopes, with f affine on
each — so a triangle mesh is extracted directly via marching cubes on a
grid of MLP evaluations. No binvox round-trip.

The single most important control is the **class-weight ratio** in the
binary cross-entropy loss, exposed here as ``bound_bias``:

- ``bound_bias < 0``: penalty on false positives grows → mesh stays
  strictly *inside* the gamut. The "inner bound" regime.
- ``bound_bias == 0``: balanced — best-fit boundary.
- ``bound_bias > 0``: penalty on false negatives grows → mesh contains
  the entire gamut plus slop. The "outer bound" regime that Liu 2024
  ships as the default.

Magnitude controls strength. ``|bound_bias|=3`` corresponds roughly to
Liu's converged ``class_weight=20``.

For farthest-color palette selection in this paper, ``bound_bias < 0``
(inner bound) is what you want — every candidate is guaranteed inside
the displayable gamut.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh

from arlabelvis.device import default_device

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralBoundingParams:
    """Hyperparameters of the in-process gamut MLP.

    The canonical paper-pipeline defaults live on ``LutConfig`` (see
    ``shape_neural_bound_bias`` and ``shape_neural_iterations``); this
    dataclass is what they get translated into at the call site. If you
    change a default here for ad-hoc experiments, it will *not* propagate
    to cached LUTs unless ``LutConfig`` is updated to match.
    """
    hidden_width: int = 50          # Liu 2024 default
    hidden_depth: int = 2           # 2 hidden layers, matching Liu
    iterations: int = 20_000
    batch_size: int = 4096
    learning_rate: float = 1e-4
    # Single control of bound character; see module docstring. The default
    # here matches ``LutConfig.shape_neural_bound_bias``.
    bound_bias: float = -1.0
    # Level set for mesh extraction.
    threshold: float = 0.5
    # Marching-cubes grid resolution.
    mesh_resolution: int = 64
    seed: int = 0


def _class_weights(bound_bias: float) -> tuple[float, float]:
    """Map bound_bias to (positive, negative) BCE class weights.

    bias=0 → (1, 1). |bias|=3 → ratio of ~20 (matches Liu's converged value).
    Sign flips which side is penalised.
    """
    strength = math.exp(abs(bound_bias) * math.log(20) / 3.0)  # 1..20 as |bias|: 0..3
    if bound_bias < 0:
        return 1.0, strength       # neg_weight high → forbid FP → inner
    if bound_bias > 0:
        return strength, 1.0       # pos_weight high → forbid FN → outer
    return 1.0, 1.0


class _GamutMLP(nn.Module):
    def __init__(self, hidden_width: int, hidden_depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_width), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_width, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 3) -> (N,)
        return self.net(x).squeeze(-1)


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor,
                  pos_weight: float, neg_weight: float) -> torch.Tensor:
    """Class-weighted BCE matching the external submodule's loss formula."""
    eps = 1e-7
    pos = target * torch.log(pred + eps).clamp(min=-100.0) * pos_weight
    neg = (1.0 - target) * torch.log(1.0 - pred + eps).clamp(min=-100.0) * neg_weight
    return -(pos + neg).mean()


@dataclass(frozen=True)
class _AffineMap:
    """Affine pre-conditioner: ``normalised = (x - center) / half_span``.

    Maps the padded working-space bbox into the symmetric cube ``[-1, 1]³``.
    Centred on zero so the first ReLU layer sees a balanced sign distribution
    on its pre-activations — all-positive inputs are a known footgun for
    dead-ReLU stickiness. The composite ``MLP(normalise(x))`` is still a
    piecewise-linear function of working-space ``x`` (the polytopes are
    just sheared by the affine), so the geometric character of the bound
    is preserved while the optimiser sees well-conditioned inputs regardless
    of which working space we're in (CIELAB ~100, OKLAB ~0.4, sRGB ~255).
    """
    center: np.ndarray   # (3,)
    half_span: np.ndarray  # (3,) — half the padded-bbox extent per axis

    @classmethod
    def for_points(cls, points: np.ndarray, pad_frac: float = 0.05) -> "_AffineMap":
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = maxs - mins
        pad = pad_frac * np.where(span > 0, span, 1.0)
        bbox_lo = mins - pad
        bbox_hi = maxs + pad
        center = 0.5 * (bbox_lo + bbox_hi)
        half_span = 0.5 * np.where(bbox_hi - bbox_lo > 0, bbox_hi - bbox_lo, 1.0)
        return cls(center=center.astype(np.float32),
                   half_span=half_span.astype(np.float32))

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        """Map working-space points → ``[-1, 1]³``."""
        return (x - self.center) / self.half_span

    def from_unit(self, x: np.ndarray) -> np.ndarray:
        """Inverse: ``[-1, 1]³`` → working space."""
        return x * self.half_span + self.center


def train_gamut_mlp(gamut_points: np.ndarray,
                    params: NeuralBoundingParams = NeuralBoundingParams(),
                    device: Optional[str] = None,
                    ) -> tuple[_GamutMLP, _AffineMap]:
    """Train a small MLP on the gamut indicator.

    Inputs are pre-conditioned by an affine map into the symmetric cube
    ``[-1, 1]³`` before the MLP — see ``_AffineMap`` for why. The composite
    ``MLP ∘ affine`` remains a piecewise-linear function of working-space
    coordinates, so the geometric form is preserved.

    Positives are drawn from ``gamut_points``. Negatives are uniform samples
    from ``[-1, 1]³`` in normalised space; on average a fraction will fall
    outside the gamut, providing the negative class without an explicit
    boundary detector.

    Returns ``(mlp, affine)`` for downstream mesh extraction in the same
    coordinate frame.
    """
    if device is None:
        device = default_device()
    pos_w, neg_w = _class_weights(params.bound_bias)
    _log.info("neural_bounding: bound_bias=%+.2f → pos_weight=%.2f, neg_weight=%.2f (device=%s)",
              params.bound_bias, pos_w, neg_w, device)

    torch.manual_seed(params.seed)

    affine = _AffineMap.for_points(gamut_points, pad_frac=0.05)
    pts_unit = affine.to_unit(gamut_points.astype(np.float32))
    pts_t = torch.from_numpy(pts_unit).to(device)
    n_pos = len(pts_t)

    mlp = _GamutMLP(params.hidden_width, params.hidden_depth).to(device)
    opt = optim.Adam(mlp.parameters(), lr=params.learning_rate)

    half = params.batch_size // 2
    for it in range(params.iterations):
        # Positives: random sample from the gamut points (in normalised coords).
        pos_idx = torch.randint(0, n_pos, (half,), device=device)
        pos_batch = pts_t[pos_idx]
        # Negatives: uniform in [-1, 1]^3.
        neg_batch = torch.rand(half, 3, device=device) * 2.0 - 1.0

        x = torch.cat([pos_batch, neg_batch], dim=0)
        y = torch.cat([torch.ones(half, device=device),
                       torch.zeros(half, device=device)])
        loss = _weighted_bce(mlp(x), y, pos_w, neg_w)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (it + 1) % max(1, params.iterations // 10) == 0:
            _log.info("    iter %d/%d  loss=%.4f", it + 1, params.iterations, loss.item())

    return mlp, affine


def _mlp_affine_per_layer(mlp: _GamutMLP) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Unpack the MLP into a list of (W, b, has_relu) triples per linear layer.

    The trailing sigmoid is dropped (we work in pre-sigmoid logit space).
    For each ReLU-followed Linear layer ``has_relu=True``; for the final
    output linear (no ReLU after it, sigmoid handled externally) ``False``.
    """
    triples: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seq = list(mlp.net.children())
    i = 0
    while i < len(seq):
        layer = seq[i]
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            has_relu = (i + 1 < len(seq) and isinstance(seq[i + 1], nn.ReLU))
            triples.append((W, b, has_relu))
            i += 2 if has_relu else 1
        else:
            i += 1
    return triples


def _affine_for_pattern(layers: list[tuple[np.ndarray, np.ndarray, bool]],
                        pattern: tuple[np.ndarray, ...]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Compose the linear layers under a fixed activation pattern.

    Returns ``(u, v)`` with ``u ∈ R^3``, ``v ∈ R`` such that the MLP's
    pre-sigmoid logit equals ``u·x + v`` for all ``x`` whose activation
    pattern matches ``pattern``. Each entry of ``pattern`` is a 0/1 mask
    over the ReLU outputs of one layer.
    """
    A = np.eye(3, dtype=np.float64)
    c = np.zeros(3, dtype=np.float64)
    pat_idx = 0
    for W, b, has_relu in layers:
        # pre-activation: W @ (A x + c) + b = (W A) x + (W c + b)
        A = W @ A
        c = W @ c + b
        if has_relu:
            mask = pattern[pat_idx].astype(np.float64)
            A = mask[:, None] * A
            c = mask * c
            pat_idx += 1
    return A.ravel(), float(c.item() if c.size == 1 else c[0])


def _affine_layer_pre_acts(layers: list[tuple[np.ndarray, np.ndarray, bool]],
                            pattern: tuple[np.ndarray, ...],
                            up_to_layer: int
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Affine map for the pre-activations of layer ``up_to_layer`` under
    the given pattern. Returns ``(C, d)`` with ``C ∈ R^{N×3}``, ``d ∈ R^N``
    so pre-act_i(x) = C[i]·x + d[i]."""
    A = np.eye(3, dtype=np.float64)
    c = np.zeros(3, dtype=np.float64)
    pat_idx = 0
    for li, (W, b, has_relu) in enumerate(layers):
        # Pre-activations BEFORE applying this layer's ReLU:
        Pre_C = W @ A
        Pre_d = W @ c + b
        if li == up_to_layer:
            return Pre_C, Pre_d
        if has_relu:
            mask = pattern[pat_idx].astype(np.float64)
            A = mask[:, None] * Pre_C
            c = mask * Pre_d
            pat_idx += 1
        else:
            A, c = Pre_C, Pre_d
    raise IndexError(f"up_to_layer={up_to_layer} out of range")


def _patterns_at_points(mlp: _GamutMLP, points: np.ndarray) -> tuple[np.ndarray, ...]:
    """Activation pattern (one mask per ReLU-followed layer) at each point.

    Returns a tuple of ``(N, hidden_width_l)`` boolean arrays — one per
    hidden ReLU layer.
    """
    masks: list[np.ndarray] = []
    h = torch.from_numpy(points.astype(np.float32))
    with torch.no_grad():
        for layer in mlp.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
            elif isinstance(layer, nn.ReLU):
                masks.append((h > 0).numpy())
                h = torch.relu(h)
            elif isinstance(layer, nn.Sigmoid):
                break
    return tuple(masks)


def _enumerate_patterns(mlp: _GamutMLP, n_samples: int = 50_000,
                        seed: int = 0) -> list[tuple[np.ndarray, ...]]:
    """Find unique activation patterns by sampling the [-1, 1]^3 cube
    uniformly. For our small MLP (50+50 ReLUs) the surface visits ≪50k
    polytopes in practice."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, size=(n_samples, 3)).astype(np.float32)
    pat_layers = _patterns_at_points(mlp, pts)
    # Hash each row's full pattern to a string key for uniqueness.
    n_layers = len(pat_layers)
    seen: dict[bytes, tuple[np.ndarray, ...]] = {}
    for r in range(n_samples):
        key_parts = [pat_layers[L][r].tobytes() for L in range(n_layers)]
        key = b"|".join(key_parts)
        if key not in seen:
            seen[key] = tuple(pat_layers[L][r].copy() for L in range(n_layers))
    return list(seen.values())


def _clip_polygon_to_halfplane(polygon: np.ndarray,
                                line_a: float, line_b: float, line_c: float,
                                keep_positive: bool) -> np.ndarray:
    """Sutherland-Hodgman clip a 2D convex polygon against the half-plane
    ``a*s + b*t + c >= 0`` (if ``keep_positive``) or ``<= 0`` else.

    ``polygon`` is ``(N, 2)``; the result is ``(M, 2)`` with M >= 0.
    """
    if len(polygon) == 0:
        return polygon
    sign = 1.0 if keep_positive else -1.0
    a, b, c = sign * line_a, sign * line_b, sign * line_c
    # signed value at each vertex: positive = keep
    vals = polygon[:, 0] * a + polygon[:, 1] * b + c
    out: list[np.ndarray] = []
    n = len(polygon)
    for i in range(n):
        p_curr = polygon[i]
        p_prev = polygon[(i - 1) % n]
        v_curr = vals[i]
        v_prev = vals[(i - 1) % n]
        if v_curr >= 0:
            if v_prev < 0:
                # entering: add intersection
                t = v_prev / (v_prev - v_curr)
                out.append(p_prev + t * (p_curr - p_prev))
            out.append(p_curr)
        elif v_prev >= 0:
            # leaving: add intersection
            t = v_prev / (v_prev - v_curr)
            out.append(p_prev + t * (p_curr - p_prev))
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 2))


def _polygon_from_pattern(layers: list[tuple[np.ndarray, np.ndarray, bool]],
                          pattern: tuple[np.ndarray, ...],
                          bbox_lo: np.ndarray, bbox_hi: np.ndarray,
                          logit_threshold: float) -> np.ndarray:
    """Build the exact level-set polygon inside one polytope.

    Returns an ``(M, 3)`` array of polygon vertices in the same coordinate
    frame as ``bbox_lo``/``bbox_hi``. Empty if the level set doesn't pass
    through the polytope.
    """
    # 1. Affine of network logit in this polytope: logit(x) = u·x + v
    u, v = _affine_for_pattern(layers, pattern)
    if np.linalg.norm(u) < 1e-12:
        return np.empty((0, 3))

    # 2. Level-set plane: u·x = logit_threshold - v.
    # Build a 2D parameterisation: pick a basis (e1, e2) orthogonal to u.
    n = u / np.linalg.norm(u)
    # Pick the world axis least aligned with n.
    helper = np.eye(3)[np.argmin(np.abs(n))]
    e1 = helper - (helper @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    p0 = ((logit_threshold - v) / (u @ u)) * u  # closest point on plane to origin

    # 3. Initial polygon: clip the level-set plane against the bbox.
    # Bbox = 6 half-spaces; clip in 2D using each.
    # We start with a large quad in the plane and clip down. The bbox
    # diagonal is a sufficient initial radius.
    diag = float(np.linalg.norm(bbox_hi - bbox_lo))
    R = diag * 1.2
    polygon = np.array([
        [-R, -R], [R, -R], [R, R], [-R, R]
    ], dtype=np.float64)

    # Helper: map (s,t) -> 3D
    def to_3d(poly2: np.ndarray) -> np.ndarray:
        return p0[None, :] + poly2[:, [0]] * e1[None, :] + poly2[:, [1]] * e2[None, :]

    # 4. Clip against bbox half-spaces.
    # bbox: x_i >= bbox_lo[i] and x_i <= bbox_hi[i]. In (s,t):
    # x_i(s,t) = p0[i] + s*e1[i] + t*e2[i]. So s*e1[i] + t*e2[i] >= bbox_lo[i] - p0[i],
    # and s*e1[i] + t*e2[i] <= bbox_hi[i] - p0[i].
    for i in range(3):
        polygon = _clip_polygon_to_halfplane(
            polygon, e1[i], e2[i], p0[i] - bbox_lo[i], keep_positive=True
        )
        if len(polygon) == 0:
            return np.empty((0, 3))
        polygon = _clip_polygon_to_halfplane(
            polygon, e1[i], e2[i], p0[i] - bbox_hi[i], keep_positive=False
        )
        if len(polygon) == 0:
            return np.empty((0, 3))

    # 5. Clip against each ReLU constraint, in pattern order.
    for layer_idx, mask in enumerate(pattern):
        C, d = _affine_layer_pre_acts(layers, pattern, up_to_layer=layer_idx)
        # For each neuron i in this layer:
        #   pre-act_i(x) = C[i]·x + d[i]
        #   pattern bit = 1 means we need pre-act_i(x) >= 0 in this region
        #   pattern bit = 0 means pre-act_i(x) <= 0.
        # In (s,t): pre-act_i = C[i]·(p0 + s*e1 + t*e2) + d[i]
        #         = (C[i]·e1) s + (C[i]·e2) t + (C[i]·p0 + d[i])
        for ni in range(C.shape[0]):
            line_a = float(C[ni] @ e1)
            line_b = float(C[ni] @ e2)
            line_c = float(C[ni] @ p0 + d[ni])
            keep_positive = bool(mask[ni])
            polygon = _clip_polygon_to_halfplane(
                polygon, line_a, line_b, line_c, keep_positive=keep_positive
            )
            if len(polygon) == 0:
                return np.empty((0, 3))

    return to_3d(polygon)


def mesh_from_mlp(mlp: _GamutMLP, affine: _AffineMap,
                  params: NeuralBoundingParams = NeuralBoundingParams(),
                  device: Optional[str] = None,
                  pattern_samples: int = 50_000) -> trimesh.Trimesh:
    """Extract the **exact** piecewise-linear level set of the trained ReLU
    MLP: enumerate the activation patterns the surface visits, derive each
    polytope's affine, clip the level-set plane against the polytope's
    constraints, fan-triangulate, glue.

    No marching cubes, no sampling-based reconstruction. The mesh's
    triangulation mirrors the MLP's polytope partition.

    The mesh lives in the normalised ``[-1, 1]³`` cube; ``affine.from_unit``
    maps the verts back to working space.
    """
    layers = _mlp_affine_per_layer(mlp)
    # Sigmoid level threshold → pre-sigmoid logit threshold.
    logit_thresh = float(np.log(params.threshold / (1.0 - params.threshold)))
    bbox_lo = np.array([-1.0, -1.0, -1.0])
    bbox_hi = np.array([+1.0, +1.0, +1.0])

    patterns = _enumerate_patterns(mlp, n_samples=pattern_samples,
                                    seed=params.seed)
    _log.info("mlp mesh: %d unique activation patterns sampled", len(patterns))

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0
    n_nonempty = 0
    for pattern in patterns:
        polygon = _polygon_from_pattern(layers, pattern, bbox_lo, bbox_hi,
                                        logit_thresh)
        if len(polygon) < 3:
            continue
        n_nonempty += 1
        # Fan-triangulate the convex polygon
        n = len(polygon)
        faces = np.stack([
            np.zeros(n - 2, dtype=np.int64),
            np.arange(1, n - 1, dtype=np.int64),
            np.arange(2, n, dtype=np.int64),
        ], axis=1) + vert_offset
        all_verts.append(polygon)
        all_faces.append(faces)
        vert_offset += n

    _log.info("mlp mesh: %d/%d patterns crossed by surface", n_nonempty, len(patterns))

    if not all_verts:
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))

    verts_unit = np.concatenate(all_verts, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0)
    verts_world = affine.from_unit(verts_unit)
    return trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)


def mesh_from_mlp_mc(mlp: _GamutMLP, affine: _AffineMap,
                      params: NeuralBoundingParams = NeuralBoundingParams(),
                      device: Optional[str] = None) -> trimesh.Trimesh:
    """Marching-cubes mesh extraction in normalised ``[-1, 1]³``.

    Robust fallback for ``mesh_from_mlp``: any closed level set sampled
    densely enough produces a watertight manifold, which the polytope
    extraction does not always achieve when a polytope's neighbour across
    the level set is missed by the activation-pattern sampler. Coarser
    triangulation than the polytope path but always feeds RGD cleanly.
    """
    if device is None:
        device = default_device()
    from skimage.measure import marching_cubes
    res = max(16, int(params.mesh_resolution))
    g = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    with torch.no_grad():
        vals = mlp(torch.from_numpy(pts).to(device)).cpu().numpy().reshape(res, res, res)

    if vals.min() >= params.threshold or vals.max() <= params.threshold:
        return trimesh.Trimesh(vertices=np.empty((0, 3)),
                               faces=np.empty((0, 3), dtype=np.int64))

    verts_voxel, faces, _, _ = marching_cubes(vals, level=params.threshold)
    # voxel coords [0, res-1] -> normalised [-1, 1] -> working-space
    verts_unit = (verts_voxel / (res - 1)) * 2.0 - 1.0
    verts_world = affine.from_unit(verts_unit.astype(np.float32))
    return trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)


def neural_bounded_mesh_inprocess(
    gamut_points: np.ndarray,
    params: NeuralBoundingParams = NeuralBoundingParams(),
    device: Optional[str] = None,
) -> trimesh.Trimesh:
    """End-to-end: train MLP, extract mesh. Drop-in replacement for the
    external binvox pipeline. Few-second runtime on CPU at default params.

    The MLP optimises in a centred unit-cube preconditioner; the returned
    mesh's vertices are in working-space coordinates.

    Tries the *exact* polytope-arrangement extraction first (preserves the
    piecewise-linear structure of the ReLU MLP). Falls back to marching
    cubes when the polytope path produces a non-watertight mesh — which
    can happen if the activation-pattern sampler misses a polytope across
    a level-set crossing, leaving a hole. RGD's cotangent Laplacian
    requires a manifold to factorise, so the fallback is necessary, not
    optional.
    """
    mlp, affine = train_gamut_mlp(gamut_points, params, device)
    # Move MLP to CPU for mesh extraction — polytope clipping and marching
    # cubes are numpy-based and need CPU tensors.
    mlp = mlp.cpu()
    mesh = mesh_from_mlp(mlp, affine, params, "cpu")
    if (len(mesh.faces) == 0
            or not mesh.is_watertight
            or not mesh.is_winding_consistent):
        _log.warning(
            "neural_bounding: polytope extraction produced non-manifold mesh "
            "(verts=%d faces=%d watertight=%s); falling back to marching cubes.",
            len(mesh.vertices), len(mesh.faces),
            getattr(mesh, "is_watertight", "?"),
        )
        mesh = mesh_from_mlp_mc(mlp, affine, params, "cpu")
    return mesh


# ---------------------------------------------------------------------------
# Diagnostics: how does the bound compare to the true gamut?
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundReport:
    """Summary of how a candidate mesh sits relative to the true gamut.

    Two complementary, non-convex-aware tests:

    - ``gamut_coverage``: fraction of the gamut point cloud that lies
      *inside* the mesh (using the actual mesh, not its convex hull).
      Computed via ``trimesh.contains``. High → mesh fills the gamut.
      For an *inner* bound this is the ratio "how much of the gamut
      does my inner-bound mesh actually cover?"
    - ``mean_vert_distance_to_gamut``: mean (in working-space units) of
      each mesh vertex's distance to its nearest gamut point. Low →
      mesh hugs the gamut. For an *outer* bound this is positive (mesh
      sticks out into empty space); for an *inner* bound this should
      be ~0 (every mesh vert is close to a gamut sample).
    - ``max_vert_distance_to_gamut``: worst-case version of the above.

    Reading combinations:
      coverage high + dist low  → tight inner bound (mesh inside gamut, fills it)
      coverage high + dist high → outer bound (mesh contains gamut + empty space)
      coverage low  + dist low  → small inner mesh, doesn't cover gamut
      coverage low  + dist high → bad: mesh and gamut barely overlap
    """
    n_vertices: int
    n_faces: int
    gamut_coverage: float
    mean_vert_distance_to_gamut: float
    max_vert_distance_to_gamut: float

    def __str__(self) -> str:
        return (f"verts={self.n_vertices} faces={self.n_faces}  "
                f"coverage={self.gamut_coverage:.1%}  "
                f"vert-to-gamut dist: "
                f"mean={self.mean_vert_distance_to_gamut:.2f} "
                f"max={self.max_vert_distance_to_gamut:.2f}")


def evaluate_bound(mesh: trimesh.Trimesh,
                   gamut_points: np.ndarray,
                   gamut_sample_size: int = 5_000) -> BoundReport:
    """Non-convex-aware bound diagnostic.

    - Coverage: ``trimesh.contains`` answers per-point "is this gamut point
      inside the (possibly non-convex) mesh?" The fraction TRUE is the
      mesh's gamut coverage.
    - Vertex-to-gamut distance: nearest-neighbour distance from each mesh
      vertex to the gamut point cloud, via a KDTree. Mean and max
      summarise how far the mesh sticks out into empty space.
    """
    from scipy.spatial import cKDTree

    verts = np.asarray(mesh.vertices)
    n_verts = len(verts)
    n_faces = int(len(mesh.faces))

    if n_verts == 0 or n_faces == 0:
        return BoundReport(0, 0, 0.0, float("inf"), float("inf"))

    rng = np.random.default_rng(0)
    if len(gamut_points) > gamut_sample_size:
        idx = rng.choice(len(gamut_points), size=gamut_sample_size, replace=False)
        sample = gamut_points[idx]
    else:
        sample = gamut_points

    # Gamut coverage: which sample points are inside the (non-convex) mesh?
    try:
        contained = mesh.contains(sample)
        coverage = float(np.mean(contained))
    except Exception:
        coverage = float("nan")

    # Vertex-to-gamut nearest-neighbour distances.
    tree = cKDTree(gamut_points)
    dists, _ = tree.query(verts)
    mean_d = float(dists.mean())
    max_d = float(dists.max())

    return BoundReport(
        n_vertices=n_verts,
        n_faces=n_faces,
        gamut_coverage=coverage,
        mean_vert_distance_to_gamut=mean_d,
        max_vert_distance_to_gamut=max_d,
    )


def signed_hull_distance(points: np.ndarray, hull_pts: np.ndarray) -> np.ndarray:
    """Signed distance from each ``point`` to the convex hull of ``hull_pts``.

    Negative inside, positive outside. Kept as a utility for the inspector
    figure (its green-inside / red-outside edge colouring uses this); the
    convex hull is a fast, smooth scalar field even though the gamut isn't
    actually convex.
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(hull_pts)
    eqs = hull.equations
    return np.max(points @ eqs[:, :3].T + eqs[:, 3], axis=1)
