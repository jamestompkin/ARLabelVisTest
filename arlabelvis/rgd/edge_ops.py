"""Edge-based mesh operators for the extended RGD port.

Provides the subset of MeshClass.m that lives outside mesh_ops.py:
  - `rotation_operator(Nf)`       -> R (3nf x 3nf sparse, applies cross(Nf, v))
  - `edge_basis(V, F, Nf)`        -> (F1, F2, EB, EBI)
  - `edge_adjacency(F)`           -> (edges, e2t, ie, inner_edges)
  - `edge_areas(F, ta, edges, e2t, ie)` -> (ne,)
  - `godf(V, F, Nf, F1, F2, ta, edges, e2t, ie, inner_edges, n)` ->
        (op, oph) where op = oph^T oph is the globally-optimal-direction-field
        smoothness operator (2nf x 2nf sparse).

Everything matches `external/matlab_rgd/MeshClass.m` conventions.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, vstack, diags


def rotation_operator(Nf: np.ndarray) -> csr_matrix:
    """R: (3nf, 3nf) sparse.  Applied to v of shape (3nf,) where v = [x; y; z]
    (x, y, z each nf-long), produces cross(Nf_f, v_f) per face in the same layout.

    Matches `MeshClass.rot`.
    """
    nf = Nf.shape[0]
    # MATLAB II is (6 nf,) and JJ is (6 nf,). Pattern per face f:
    #   row block gets 2 copies of f, (f+nf), (f+2nf)
    #   col matches: [f+nf, f+2nf, f, f+2nf, f, f+nf]
    #   vals: [-n[2], n[1], n[2], -n[0], -n[1], n[0]]
    II = np.concatenate([np.arange(nf)]*2 + [np.arange(nf, 2*nf)]*2 + [np.arange(2*nf, 3*nf)]*2)
    JJ = np.concatenate([
        np.arange(nf, 2*nf), np.arange(2*nf, 3*nf),   # rows for x-component
        np.arange(nf),       np.arange(2*nf, 3*nf),   # rows for y-component
        np.arange(nf),       np.arange(nf, 2*nf),     # rows for z-component
    ])
    SS = np.concatenate([
        -Nf[:, 2],  Nf[:, 1],
         Nf[:, 2], -Nf[:, 0],
        -Nf[:, 1],  Nf[:, 0],
    ])
    return csr_matrix((SS, (II, JJ)), shape=(3*nf, 3*nf))


def edge_basis(V: np.ndarray, F: np.ndarray, Nf: np.ndarray):
    """Per-face tangent-plane basis and the 3D<->2D projection matrices.

    Returns:
        F1 (nf, 3)      First basis vector per face (normalized E1).
        F2 (nf, 3)      Second basis vector = cross(Nf, F1), unit-norm.
        EB (2 nf, 3 nf) sparse, maps 3D face vectors (stacked [x;y;z]) to 2D tangent coords.
        EBI (3 nf, 2 nf) sparse, EB.T (the pseudo-inverse / adjoint map).

    Matches `MeshClass.edge_basis`.
    """
    nf = F.shape[0]
    E1 = V[F[:, 1]] - V[F[:, 2]]
    lens = np.linalg.norm(E1, axis=1, keepdims=True)
    F1 = E1 / lens
    F2 = np.cross(Nf, F1)

    # MATLAB:
    #   I = [1..nf, 1..nf, 1..nf]
    #   J = [1..nf, nf+1..2nf, 2nf+1..3nf]
    #   B1 values = NE1(:) = [NE1x; NE1y; NE1z]  (column stack)
    rows = np.tile(np.arange(nf), 3)
    cols = np.concatenate([np.arange(nf), np.arange(nf) + nf, np.arange(nf) + 2*nf])
    vals1 = np.concatenate([F1[:, 0], F1[:, 1], F1[:, 2]])
    vals2 = np.concatenate([F2[:, 0], F2[:, 1], F2[:, 2]])
    B1 = csr_matrix((vals1, (rows, cols)), shape=(nf, 3*nf))
    B2 = csr_matrix((vals2, (rows, cols)), shape=(nf, 3*nf))
    EB = vstack([B1, B2]).tocsr()
    EBI = EB.T.tocsr()
    return F1, F2, EB, EBI


def edge_adjacency(F: np.ndarray):
    """Edge enumeration + edge-face adjacency.

    Returns:
        edges (ne, 2)       Sorted vertex index pairs (smaller vertex first).
        e2t (ne, 2)         Face indices incident to each edge. Second column is
                            -1 for boundary edges.
        ie (ne,) bool       True for interior edges (two incident faces).
        inner_edges (nie,)  Indices of interior edges.

    Simpler than MATLAB's `nc_data` because we don't need signed ordering or
    vertex-edge sparse mappings — `godf` only needs faces-per-edge and vertex
    pair per interior edge.
    """
    nf = F.shape[0]
    # Collect all 3 edges per face
    edges_all = np.vstack([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]])  # (3 nf, 2)
    faces_of_edge = np.tile(np.arange(nf), 3)  # (3 nf,)
    # Canonicalize: lower vertex first
    edges_sorted = np.sort(edges_all, axis=1)
    # Group by edge — use a structured key
    keys = edges_sorted[:, 0].astype(np.int64) * (F.max() + 2) + edges_sorted[:, 1]
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    faces_sorted = faces_of_edge[order]
    edges_canon = edges_sorted[order]

    # Find unique keys (edges) and the run boundaries
    unique_mask = np.concatenate([[True], keys_sorted[1:] != keys_sorted[:-1]])
    first = np.where(unique_mask)[0]
    ne = len(first)
    edges = edges_canon[first]
    counts = np.diff(np.concatenate([first, [len(keys_sorted)]]))

    e2t = np.full((ne, 2), -1, dtype=np.int64)
    for k, (start, cnt) in enumerate(zip(first, counts)):
        if cnt >= 1:
            e2t[k, 0] = faces_sorted[start]
        if cnt >= 2:
            e2t[k, 1] = faces_sorted[start + 1]
        # Meshes with non-manifold edges (cnt > 2) are rare; we'd need to handle
        # them explicitly, but the MATLAB version assumes manifoldness too.
    ie = (counts == 2)
    inner_edges = np.where(ie)[0]
    return edges, e2t, ie, inner_edges


def edge_areas(ta: np.ndarray, e2t: np.ndarray, ie: np.ndarray) -> np.ndarray:
    """Edge areas matching `MeshClass.edge_areas`.

    For a matched interior edge e with incident faces (f1, f2):
        ea[e] = 2/3 * (ta[f1] + ta[f2])
    For boundary edges the MATLAB code adds only one face's contribution twice
    (identity both directions collapse), yielding 2/3 * ta[f1].

    We derive directly from the face areas rather than reconstructing the
    MATLAB sparse `W` helper.
    """
    ne = e2t.shape[0]
    ea = np.zeros(ne, dtype=float)
    f1 = e2t[:, 0]
    f2 = e2t[:, 1]
    # Interior: both incident faces
    both = ie
    ea[both] = (2.0 / 3.0) * (ta[f1[both]] + ta[f2[both]])
    # Boundary: only f1 is set (f2 == -1). MATLAB's formula reduces to 2/3 * ta[f1].
    bnd = ~ie
    ea[bnd] = (2.0 / 3.0) * ta[f1[bnd]]
    return ea


def godf(V: np.ndarray, F: np.ndarray, Nf: np.ndarray,
         F1: np.ndarray, F2: np.ndarray,
         edges: np.ndarray, e2t: np.ndarray, ie: np.ndarray, inner_edges: np.ndarray,
         ea: np.ndarray, n: int):
    """Globally-optimal-direction-field smoothness operator.

    Returns (op, oph) with op = oph^T @ oph.
    `op` is (2 nf, 2 nf); `oph` is (2 nie, 2 nf).

    Matches `MeshClass.godf`.
    """
    nf = F.shape[0]
    nie = len(inner_edges)

    t1 = e2t[inner_edges, 0]
    t2 = e2t[inner_edges, 1]

    # Edge vector v2 -> v1 (MATLAB does edges(:,2) - edges(:,1))
    EV = V[edges[inner_edges, 1]] - V[edges[inner_edges, 0]]
    EV = EV / np.linalg.norm(EV, axis=1, keepdims=True)

    # Angle of EV in each incident face's tangent basis (F1, F2)
    IN1 = np.arctan2(np.sum(EV * F2[t1], axis=1), np.sum(EV * F1[t1], axis=1))
    IN2 = np.arctan2(np.sum(EV * F2[t2], axis=1), np.sum(EV * F1[t2], axis=1))
    PT = n * (IN2 - IN1)

    # Build 2 nie x 2 nf CovD sparse. Block pattern per inner edge e:
    #   row  e:         cos(PT) at (e, t1),   -sin(PT) at (e, t1+nf),  -1 at (e, t2),   0 at (e, t2+nf)
    #   row  e+nie:     sin(PT) at (e+nie, t1), cos(PT) at (e+nie, t1+nf), 0 at (e+nie, t2), -1 at (e+nie, t2+nf)
    II = np.concatenate([
        np.arange(nie),            # block 1 row entries
        np.arange(nie),
        np.arange(nie),
        np.arange(nie),
        np.arange(nie) + nie,      # block 2 row entries
        np.arange(nie) + nie,
        np.arange(nie) + nie,
        np.arange(nie) + nie,
    ])
    JJ = np.concatenate([
        t1,
        t1 + nf,
        t2,
        t2 + nf,
        t1,
        t1 + nf,
        t2,
        t2 + nf,
    ])
    SS = np.concatenate([
         np.cos(PT),  -np.sin(PT),  -np.ones(nie),  np.zeros(nie),
         np.sin(PT),   np.cos(PT),   np.zeros(nie), -np.ones(nie),
    ])
    CovD = csr_matrix((SS, (II, JJ)), shape=(2 * nie, 2 * nf))

    # Edge-area weighting (sqrt of edge areas, interior only)
    sqrt_ea_inner = np.sqrt(ea[inner_edges])
    Ws = diags(np.tile(sqrt_ea_inner, 2))
    oph = (Ws @ CovD).tocsr()
    op = (oph.T @ oph).tocsr()
    return op, oph
