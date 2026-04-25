"""Output-path helpers for the short paper.

The short paper's figures and tables land in the Overleaf project that
contains ``ieeevis2026/paper.tex``. The actual location of that checkout
varies per contributor; set ``$ARLABELVIS_SHORTPAPER_DIR`` to the
absolute path of the ``ieeevis2026/`` directory. If unset, outputs land in
``<repo>/results/shortpaper/`` so scripts are runnable from any clone
without env-var configuration (results just won't appear in Overleaf).
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SHORTPAPER_DIR = _REPO_ROOT / "results" / "shortpaper"
SHORTPAPER_DIR = Path(
    os.environ.get("ARLABELVIS_SHORTPAPER_DIR", str(_DEFAULT_SHORTPAPER_DIR))
)
SHORTPAPER_FIG_DIR = SHORTPAPER_DIR / "figures"
SHORTPAPER_TAB_DIR = SHORTPAPER_DIR / "tables"


def fig_path(subdir: str, name: str) -> Path:
    """Figure output path. ``subdir`` is the figure label (without the ``fig:``
    prefix) — e.g. ``teaser``, ``space_comparison`` — so on-disk grouping
    matches the LaTeX label structure. ``name`` should include the extension."""
    target = SHORTPAPER_FIG_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target / name


def tab_path(subdir: str, name: str) -> Path:
    """Table output path. Same pattern as ``fig_path`` but rooted at
    ``tables/``. Use for ``.tex`` fragments that ``paper.tex`` ``\\input{}``s."""
    target = SHORTPAPER_TAB_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target / name
