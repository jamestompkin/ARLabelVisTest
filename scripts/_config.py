"""Shared run configuration + filename conventions used across scripts.

The hardcoded paths mirror what the old monolithic `main.py` produced.
Edit a script's `CONFIG = RunConfig(...)` block to change parameters per run.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    interval: int = 1                  # stepSize when sampling the 256^3 RGB grid
    space: str = "CIELAB"              # "CIELAB" | "OKLAB" | "RGB"
    distance_measure: str = "RGD"      # "RGD" | "Euclidean"
    alpha: str = "75"                  # alpha_hat*100 as a stringy tag (matches existing filenames)
    voxel_dim: int = 256               # neural bounding voxel resolution
    sigma: float = 0.0                 # Gaussian-filter sigma (only used when smoothing_mode == "none")
    vox: int = 256                     # voxel resolution for meshing
    smoothing_mode: str = "neural"     # "none" | "sphere" | "neural" | "pytorch"

    @property
    def str_sigma(self) -> str:
        return str(self.sigma).replace(".", "o")

    @property
    def original_off_path(self) -> str:
        if self.smoothing_mode == "none":
            return f"external/matlab_rgd/RGB2{self.space}_{self.interval}_sigma_{self.str_sigma}_vox_{self.vox}.off"
        return f"external/matlab_rgd/RGB2{self.space}_{self.smoothing_mode}_{self.interval}.off"

    @property
    def smoothed_off_path(self) -> str:
        return f"external/matlab_rgd/RGB2{self.space}_{self.smoothing_mode}_{self.interval}_sigma_{self.sigma}.off"

    @property
    def matlab_indices_path(self) -> str:
        if self.smoothing_mode == "none":
            return (f"external/matlab_rgd/max_indices_{self.space}_{self.interval}_{self.distance_measure}"
                    f"_{self.alpha}_sigma_{self.str_sigma}_vox_{self.vox}.txt")
        return (f"external/matlab_rgd/max_indices_{self.space}_{self.interval}_{self.distance_measure}"
                f"_{self.alpha}_{self.smoothing_mode}_{self.voxel_dim}.txt")

    @property
    def final_lab_path(self) -> str:
        if self.smoothing_mode == "none":
            return (f"AllCandidateLABvals_{self.space}_{self.interval}_{self.distance_measure}"
                    f"_{self.alpha}_sigma_{self.str_sigma}_vox_{self.vox}.txt")
        return (f"AllCandidateLABvals_{self.space}_{self.interval}_{self.distance_measure}"
                f"_{self.alpha}_{self.smoothing_mode}_{self.voxel_dim}.txt")

    @property
    def candidate_lab_file(self) -> str:
        return f"CandidateLABvals_MATLAB_{self.space}_{self.interval}_{self.distance_measure}_{self.alpha}.txt"

    @property
    def candidate_rgb_file(self) -> str:
        return f"CorrespondingRGBVals_MATLAB_{self.space}_{self.interval}_{self.distance_measure}_{self.alpha}.txt"

    @property
    def furthest_save_path(self) -> str:
        return f"data/MATLAB_FurthestRGB_From{self.space}_{self.interval}_{self.distance_measure}_{self.alpha}.txt"

    @property
    def binvox_path(self) -> str:
        return f"external/neural_bounding/data/3D/{self.space}_{self.interval}_{self.voxel_dim}.binvox"

    @property
    def neural_bounded_path(self) -> str:
        return f"data/neural_bounding_{self.space}_{self.voxel_dim}.binvox"

    def get_points(self):
        """Generate the input (allRGBs, allPoints-in-space) pair for this config."""
        from arlabelvis.meshing import generate_LABs
        from arlabelvis.colors import sRGBtoOKLAB
        allRGBs, allLABs = generate_LABs(stepSize=self.interval)
        if self.space == "OKLAB":
            points = sRGBtoOKLAB(allRGBs)
        elif self.space == "sRGB":
            points = allRGBs
        else:
            points = allLABs
        return allRGBs, allLABs, points
