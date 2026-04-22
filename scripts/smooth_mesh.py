"""Apply the configured smoothing method to the input color space's boundary mesh
and save the result as OFF (for consumption by external/matlab_rgd/demo.m).

Smoothing methods: 'sphere' (alpha-shape -> inscribed sphere),
'neural' (reads a pre-trained binvox from neural_bounding),
'pytorch' (differentiable mesh optimizer, see arlabelvis/meshing.py),
'none' (no smoothing — pass through).
(Was `Mode.SMOOTH_MESH` in the old main.py.)
"""
from arlabelvis.meshing import pointsToMesh, optimize_mesh
from arlabelvis.bounding import (
    bindLABtoSphere, bindToNeuralBounding, bindToOptimizedMeshBinding,
)
from arlabelvis.viz import show_original_mesh_pyvista
from arlabelvis.off import save_off_file

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", sigma=0.0, vox=256,
                   smoothing_mode="neural", voxel_dim=256)


def main():
    cfg = CONFIG
    allRGBs, _, points = cfg.get_points()

    if cfg.smoothing_mode == "sphere":
        bound_points = bindLABtoSphere(points, allRGBs)
    elif cfg.smoothing_mode == "neural":
        bound_points = bindToNeuralBounding(cfg.neural_bounded_path, cfg.voxel_dim, points, allRGBs)
    elif cfg.smoothing_mode == "pytorch":
        mesh = pointsToMesh(points, sigma=cfg.sigma, vox=cfg.vox)
        show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)
        optimized = optimize_mesh(
            mesh, n_iters=20000, lr=5e-4,
            w_smooth=1.0, w_inside=200.0, w_volume=0.5, sdf_resolution=64,
        )
        show_original_mesh_pyvista(optimized, show_edges=True, show_normals=False)
        bound_points = bindToOptimizedMeshBinding(optimized, points)
    else:  # "none"
        bound_points = points

    mesh = pointsToMesh(bound_points)
    save_off_file(cfg.smoothed_off_path, mesh)
    show_original_mesh_pyvista(mesh, show_edges=True, show_normals=False)
    print(f"saved smoothed mesh to {cfg.smoothed_off_path}")


if __name__ == "__main__":
    main()
