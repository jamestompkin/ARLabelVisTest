"""Regularized geodesic distances — Python port of Edelstein et al. 2023.

Structured so that `mesh_ops` provides the operators exactly matching MATLAB's
`MeshClass`, and `admm` provides the `rdg_ADMM` solver on top of them.
"""
