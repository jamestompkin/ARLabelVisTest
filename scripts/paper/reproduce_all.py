"""Run every registered figure/table script.

Usage:
  uv run python -m scripts.paper.reproduce_all                # run all
  uv run python -m scripts.paper.reproduce_all --force        # don't skip existing
  uv run python -m scripts.paper.reproduce_all --only fig_hue_histograms tab_geometry_smoothing
"""
from __future__ import annotations

import argparse
import importlib
import time
import traceback
from pathlib import Path

from scripts.paper._configs import FIGURES, TABLES

ROOT = Path(__file__).resolve().parents[2]


def _outputs_exist(entry) -> bool:
    return all((ROOT / o).exists() for o in entry["outputs"])


def _run_one(fig_id: str, entry: dict, force: bool) -> tuple[bool, float, str]:
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
        return False, dt, f"FAILED after {dt:.1f}s: {type(e).__name__}: {e}\n" + traceback.format_exc()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-run scripts whose outputs already exist")
    p.add_argument("--only", nargs="+", default=None, help="run only these IDs")
    p.add_argument("--skip-tables", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    entries = {}
    if not args.skip_figures:
        entries.update(FIGURES)
    if not args.skip_tables:
        entries.update(TABLES)
    if args.only:
        entries = {k: v for k, v in entries.items() if k in args.only}
        missing = [k for k in args.only if k not in entries]
        if missing:
            raise SystemExit(f"unknown IDs: {missing}")

    results = []
    for fig_id, entry in entries.items():
        print(f"\n===== {fig_id} =====")
        ok, dt, msg = _run_one(fig_id, entry, args.force)
        print(msg if ok else msg[:2000])
        results.append((fig_id, ok, dt, msg))

    print("\n===== SUMMARY =====")
    for fig_id, ok, dt, msg in results:
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {dt:6.1f}s   {fig_id}")
    n_fail = sum(1 for _, ok, *_ in results if not ok)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
