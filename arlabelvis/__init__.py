"""ARLabelVis — library of methods for lookup-table construction, rendering, and scene processing.

Organized so that:
- colors, distances, interpolate, metrics are stateless numerical/file utilities;
- meshing, bounding, voxels, off handle geometry and I/O;
- scene provides per-video CEC + LUT-lookup primitives;
- viz provides rendering primitives for figures.

Scripts that produce artifacts live in the top-level `scripts/` package;
this package never writes files on import.
"""
