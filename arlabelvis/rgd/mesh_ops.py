"""Mesh operators matching `external/matlab_rgd/MeshClass.m`.

Only the subset used by `rdg_ADMM` (Dirichlet regularizer) is implemented:
face normals, face/vertex areas, cotangent Laplacian, gradient operator.

Conventions (matched to MATLAB):
  - Faces are triangles, stored as (nf, 3) with 0-indexed vertex indices.
    MATLAB stores 1-indexed; the caller converts at I/O.
  - Face normal: `cross(v0 - v1, v0 - v2)` normalized. Matches MATLAB line-for-line.
  - Vertex areas: barycentric (each face contributes ta/3 to each of its 3 vertices).
  - Cotangent Laplacian: symmetric PSD, positive diagonal, negative off-diagonals,
    row-sums zero. Validated against MATLAB to 4e-10 on the icosphere.
  - Gradient operator G (3nf x nv):
        (G u).reshape(3, nf).T[f]  = gradient of u on face f, as a 3-vector.
    Per-face contribution:
        G[f, v_a] = -cross(Nf_f, v_b - v_c) / (2 * ta_f)     (v_a opposite edge bc)
        G[f, v_b] = +cross(Nf_f, v_a - v_c) / (2 * ta_f)
        G[f, v_c] = -cross(Nf_f, v_a - v_b) / (2 * ta_f)
    with v_a,v_b,v_c = faces[f, 0], faces[f, 1], faces[f, 2].
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class MeshOps:
    """Precomputed operators for a triangle mesh, matching MATLAB MeshClass fields."""
    V: np.ndarray              # (nv, 3) float
    F: np.ndarray              # (nf, 3) int (0-indexed)
    Nf: np.ndarray             # (nf, 3) unit face normals
    ta: np.ndarray             # (nf,)  face areas
    va: np.ndarray             # (nv,)  barycentric vertex areas
    Ww: csr_matrix             # (nv, nv) cotangent Laplacian (PSD, pos diag, zero row sums)
    G: csr_matrix              # (3 nf, nv) gradient operator

    @property
    def nv(self) -> int:
        return self.V.shape[0]

    @property
    def nf(self) -> int:
        return self.F.shape[0]


def face_normals_and_areas(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals and face areas.

    Matches MATLAB MeshClass.compute_all lines 88-91:
      Nf_unnorm = cross(v0 - v1, v0 - v2)
      Nf        = Nf_unnorm / |Nf_unnorm|
      ta        = |Nf_unnorm| / 2
    """
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    Nf_unnorm = np.cross(v0 - v1, v0 - v2)
    norms = np.linalg.norm(Nf_unnorm, axis=1)
    ta = norms / 2.0
    Nf = Nf_unnorm / norms[:, None]
    return Nf, ta


def vertex_areas(F: np.ndarray, ta: np.ndarray, nv: int) -> np.ndarray:
    """Barycentric vertex areas: va[v] = sum over faces f incident to v of ta[f]/3.

    Matches MATLAB `obj.va = obj.calculatefvConnectivity()' * obj.ta / 3`, which is
    `(fvConnectivity)^T @ ta / 3` where fvConnectivity is the nf x nv face-vertex
    incidence matrix.
    """
    va = np.zeros(nv, dtype=np.float64)
    contrib = ta / 3.0
    for corner in range(3):
        np.add.at(va, F[:, corner], contrib)
    return va


def cot_laplacian(V: np.ndarray, F: np.ndarray) -> csr_matrix:
    """Cotangent Laplacian in MATLAB's sign convention (positive diagonal, negative
    off-diagonals, row-sums zero).

    Matches `external/matlab_rgd/cotLaplacian.m`.
    """
    nv = V.shape[0]
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    v0 = V[i0]; v1 = V[i1]; v2 = V[i2]
    L1 = np.linalg.norm(v1 - v2, axis=1)   # edge opposite vertex 0
    L2 = np.linalg.norm(v0 - v2, axis=1)
    L3 = np.linalg.norm(v0 - v1, axis=1)

    # Law-of-cosines angle at each vertex
    A1 = np.arccos(np.clip((L2**2 + L3**2 - L1**2) / (2 * L2 * L3), -1, 1))
    A2 = np.arccos(np.clip((L1**2 + L3**2 - L2**2) / (2 * L1 * L3), -1, 1))
    A3 = np.arccos(np.clip((L1**2 + L2**2 - L3**2) / (2 * L1 * L2), -1, 1))

    I = np.concatenate([i0, i1, i2])
    J = np.concatenate([i1, i2, i0])
    S = 0.5 / np.tan(np.concatenate([A3, A1, A2]))

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])
    return csr_matrix((Sn, (In, Jn)), shape=(nv, nv))


def gradient_operator(V: np.ndarray, F: np.ndarray, Nf: np.ndarray, ta: np.ndarray) -> csr_matrix:
    """Per-face gradient operator G: 3nf x nv, matching `MeshClass.GG` (MATLAB).

    MATLAB stacks the 3 spatial components as rows 0..nf-1 for x,
    nf..2nf-1 for y, 2nf..3nf-1 for z (so `(G u).reshape(3, nf).T` gives an
    nf x 3 vector-per-face).
    """
    nv = V.shape[0]
    nf = F.shape[0]

    E1 = V[F[:, 1]] - V[F[:, 2]]   # edge opposite vertex 0 (called `a`/`i` in the doc above)
    E2 = V[F[:, 0]] - V[F[:, 2]]   # edge opposite vertex 1 (`b`/`j`)
    E3 = V[F[:, 0]] - V[F[:, 1]]   # edge opposite vertex 2 (`c`/`k`)

    RE1 = np.cross(Nf, E1)
    RE2 = np.cross(Nf, E2)
    RE3 = np.cross(Nf, E3)

    inv_2ta = 0.5 / ta   # (nf,)

    row_indices = []
    col_indices = []
    values = []
    face_rows = np.arange(nf, dtype=np.int64)

    # Per-component (x, y, z) row block: rows f + c*nf
    for c in range(3):
        row_block = face_rows + c * nf
        # Vertex 0 contribution: -RE1
        row_indices.append(row_block)
        col_indices.append(F[:, 0])
        values.append(-RE1[:, c] * inv_2ta)
        # Vertex 1: +RE2
        row_indices.append(row_block)
        col_indices.append(F[:, 1])
        values.append(RE2[:, c] * inv_2ta)
        # Vertex 2: -RE3
        row_indices.append(row_block)
        col_indices.append(F[:, 2])
        values.append(-RE3[:, c] * inv_2ta)

    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)
    values = np.concatenate(values)
    return csr_matrix((values, (row_indices, col_indices)), shape=(3 * nf, nv))


def build_mesh_ops(V: np.ndarray, F: np.ndarray) -> MeshOps:
    """Build the full set of operators for a mesh. One-shot constructor."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    Nf, ta = face_normals_and_areas(V, F)
    va = vertex_areas(F, ta, V.shape[0])
    Ww = cot_laplacian(V, F)
    G = gradient_operator(V, F, Nf, ta)
    return MeshOps(V=V, F=F, Nf=Nf, ta=ta, va=va, Ww=Ww, G=G)
