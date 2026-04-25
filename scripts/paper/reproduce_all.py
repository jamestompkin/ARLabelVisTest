"""Run every registered figure/table script.

Usage:
  uv run python -m scripts.paper.reproduce_all             # run all
  uv run python -m scripts.paper.reproduce_all --force     # don't skip existing
  uv run python -m scripts.paper.reproduce_all --only fig_teaser tab_timing

Tables run before figures because ``tab_timing`` invalidates + rebuilds
the slowest LUTs (ΔE₀₀ at interval=1 is ~76 min) from scratch, and every
downstream figure/table hits the cache.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time
import traceback
from pathlib import Path

from scripts.paper._configs import FIGURES, TABLES
from scripts.paper._paths import SHORTPAPER_FIG_DIR, SHORTPAPER_TAB_DIR


# Known output-kind mapping. Extend this dict when new kinds show up;
# unknown extensions raise rather than silently routing to the figures dir.
_OUTPUT_ROOTS = {
    ".png": SHORTPAPER_FIG_DIR,
    ".pdf": SHORTPAPER_FIG_DIR,
    ".tex": SHORTPAPER_TAB_DIR,
}


def _resolve(out: str) -> Path:
    """Route an output path to ``figures/`` or ``tables/`` by extension."""
    root = _OUTPUT_ROOTS.get(Path(out).suffix)
    if root is None:
        raise ValueError(
            f"unknown output extension in {out!r}; extend _OUTPUT_ROOTS "
            f"in reproduce_all.py to add a new figure or table kind."
        )
    return root / out


def _outputs_exist(entry: dict) -> bool:
    return all(_resolve(o).exists() for o in entry["outputs"])


def _run_one(entry: dict, force: bool) -> tuple[bool, float, str]:
    if not force and _outputs_exist(entry):
        return True, 0.0, "skipped (outputs exist)"
    t0 = time.perf_counter()
    try:
        mod = importlib.import_module(entry["module"])
        mod.main()
        dt = time.perf_counter() - t0
        return True, dt, f"ok in {dt:.1f}s"
    except Exception as e:
        dt = time.perf_counter() - t0
        return False, dt, (f"FAILED after {dt:.1f}s: {type(e).__name__}: {e}\n"
                           + traceback.format_exc())


def main():
    # Library code logs at INFO through arlabelvis.luts; route it to stdout with
    # no prefix so output matches the old print-based behaviour.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--force", action="store_true",
                   help="re-run scripts whose outputs already exist")
    p.add_argument("--only", nargs="+", default=None,
                   help="run only these IDs")
    args = p.parse_args()

    entries: dict[str, dict] = {}
    entries.update(TABLES)     # tables first (cache-warming order)
    entries.update(FIGURES)
    if args.only:
        missing = [k for k in args.only if k not in entries]
        if missing:
            raise SystemExit(f"unknown IDs: {missing}")
        entries = {k: v for k, v in entries.items() if k in args.only}

    results = []
    for fig_id, entry in entries.items():
        print(f"\n===== {fig_id} =====")
        ok, dt, msg = _run_one(entry, args.force)
        print(msg if ok else msg[:2000])
        results.append((fig_id, ok, dt, msg))

    print("\n===== SUMMARY =====")
    for fig_id, ok, dt, _msg in results:
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {dt:6.1f}s   {fig_id}")
    n_fail = sum(1 for _, ok, *_ in results if not ok)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
