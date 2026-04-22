"""Write the input color-space point cloud as a binvox file (input for the
neural_bounding submodule's trainer). (Was `Mode.TO_VOXELS` in the old main.py.)"""
from arlabelvis.voxels import writeVoxels

from scripts._config import RunConfig


CONFIG = RunConfig(interval=1, space="CIELAB", voxel_dim=256)


def main():
    _, _, points = CONFIG.get_points()
    writeVoxels(points, CONFIG.voxel_dim, CONFIG.binvox_path)
    print(f"wrote {CONFIG.binvox_path}")


if __name__ == "__main__":
    main()
