"""Gamut helpers used by the LUT pipeline.

After the in-process MLP path replaced the external binvox neural-bounding
pipeline, the only function this module still exports is the
``bind_lab_to_sphere`` helper used by ``_sphere_candidates`` in
``arlabelvis.luts``.
"""
from __future__ import annotations

import logging
import math

import alphashape
import numpy as np

_log = logging.getLogger(__name__)


def bind_lab_to_sphere(all_lab_points: np.ndarray,
                       all_rgb: np.ndarray) -> np.ndarray:
    """Project every lab point onto the largest sphere fully contained in
    the gamut alpha-shape.

    Used by the ``shape='sphere'`` candidate path: the resulting sphere
    radius drives an icosphere whose vertices become the candidate set.
    """
    center = np.mean(all_lab_points, axis=0)

    shape = alphashape.alphashape(all_lab_points, alpha=0.005)
    boundary = shape.vertices
    bounded_distance = math.inf
    for point in boundary:
        d = math.sqrt((point[0] - center[0]) ** 2
                      + (point[1] - center[1]) ** 2
                      + (point[2] - center[2]) ** 2)
        if d < bounded_distance:
            bounded_distance = d
    _log.info("bind_lab_to_sphere: center=%s radius=%.3f",
              np.round(center, 2).tolist(), bounded_distance)

    moved = 0
    for i in range(len(all_lab_points)):
        point = all_lab_points[i]
        d = math.sqrt((point[0] - center[0]) ** 2
                      + (point[1] - center[1]) ** 2
                      + (point[2] - center[2]) ** 2)
        if d > bounded_distance:
            move = d - bounded_distance
            direction = (point - center) / np.linalg.norm(point - center)
            all_lab_points[i] = point - move * direction
            moved += 1
    _log.info("bind_lab_to_sphere: moved %d/%d points inward to the sphere",
              moved, len(all_lab_points))
    return all_lab_points
