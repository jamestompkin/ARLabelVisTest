"""Shared helpers for short-paper (ieeevis2026) figure scripts.

Short-paper figures live at SHORTPAPER_FIG_DIR (the Overleaf project's
``ieeevis2026/figures`` directory) under whatever filenames the short paper's
``\\includegraphics{...}`` calls expect. They use the same LUT cache as the
thesis-style figures under `scripts.paper.*`.

Never write into the thesis ``figures/`` directory (see CLAUDE.md at the
Overleaf root) — that directory is read-only.
"""
from __future__ import annotations

from pathlib import Path

from scripts.paper._shared import SHORTPAPER_FIG_DIR


def shortpaper_path(subdir: str, name: str) -> Path:
    """Resolve a short-paper figure filename to its absolute Overleaf path,
    creating the parent directory if needed.

    ``subdir`` is the short-paper figure label (without the ``fig:`` prefix):
    e.g. ``teaser``, ``space_comparison``, ``smoothing``, ``hue_histograms``,
    ``alpha_plots``. Keeps the Overleaf ``figures/`` tree organised one
    directory per figure so files match their LaTeX figure labels.
    ``name`` should include the extension (e.g. ``old_lookup.png``).
    """
    target = SHORTPAPER_FIG_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target / name
