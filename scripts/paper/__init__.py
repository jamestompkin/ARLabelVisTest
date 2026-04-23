"""Paper reproducibility pipeline.

One script per figure/table. Shared infrastructure:
  - `_configs.py`   registry of (figure_id, LUT config) tuples
  - `_lut_cache.py` content-addressable LUT cache to avoid regenerating
  - `reproduce_all.py`  runs every figure+table script in order, skips
                        scripts whose outputs already exist

Run any one figure:  `uv run python -m scripts.paper.fig_srgb_euclidean`
Run everything:      `uv run python -m scripts.paper.reproduce_all`
"""
