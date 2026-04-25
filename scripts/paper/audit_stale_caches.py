"""Audit which on-disk LUT cache entries are stale after the 2026-04-25
``srgb_to_lab`` matrix fix.

Background: the sRGB→XYZ matrix in ``arlabelvis.colors`` was applied
with the wrong orientation for years. White returned L* ≈ 107 instead
of 100; red went to (66, 24, 32) instead of (53, 80, 67). The fix
landed 2026-04-25 (commit / file mtime — see git log).

Every LUT with ``working_space='CIELAB'`` whose cache file was written
*before* that date now holds numerically-incorrect outputs. Cache keys
are derived from ``LutConfig`` only (not from code), so a naive
``get_lut(cfg)`` happily returns the stale LUT.

OKLAB- and sRGB-working-space LUTs do *not* use ``srgb_to_lab`` and
are unaffected.

This script:

- Walks every ``data/luts/cache/*.meta.json``.
- Decodes the config to determine ``working_space``.
- Compares the cache file's mtime to the cutoff.
- Prints a one-line report per stale entry, plus a summary.

Pass ``--delete`` to actually remove the stale cache + meta files.
Without that, the script just reports.

Usage::

    uv run python -m scripts.paper.audit_stale_caches              # report
    uv run python -m scripts.paper.audit_stale_caches --delete     # nuke
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from arlabelvis.luts import DEFAULT_CACHE


# Anything older than this is suspect. Set to the wall-clock moment the
# matrix fix landed; padding by an hour to absorb timezone fuzz.
_CUTOFF = datetime(2026, 4, 25, 0, 0).timestamp()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delete", action="store_true",
                   help="remove the stale .npy and .meta.json files")
    p.add_argument("--cutoff", type=str, default=None,
                   help="ISO datetime override (default: 2026-04-25 00:00 local)")
    args = p.parse_args()

    cutoff = (datetime.fromisoformat(args.cutoff).timestamp()
              if args.cutoff else _CUTOFF)
    cutoff_str = datetime.fromtimestamp(cutoff).isoformat(timespec="seconds")

    cache_dir = DEFAULT_CACHE.root
    metas = sorted(cache_dir.glob("*.meta.json"))
    if not metas:
        print(f"no cache entries under {cache_dir}")
        return 0

    def _npy_for(meta: Path) -> Path:
        # ``meta.with_suffix(".npy")`` would give ``foo.meta.npy``; want ``foo.npy``.
        key = meta.name[: -len(".meta.json")]
        return meta.parent / f"{key}.npy"

    stale: list[Path] = []
    schema_stale: list[Path] = []
    fresh = 0
    unaffected = 0
    for meta in metas:
        npy = _npy_for(meta)
        if not npy.exists():
            continue
        cfg = json.loads(meta.read_text())
        # Schema-stale: missing one of the fields current LutConfig has.
        # Pre-rename cache entries used ``space`` instead of
        # ``working_space``; pre-2026-04-25 entries lack ``interp_method``
        # because that field was added when the dense bake-out switched
        # from trilinear (which silently blended algorithmic picks) to
        # nearest-neighbour. Either way, no current LutConfig will hit
        # them — they're dead disk weight.
        if "working_space" not in cfg or "interp_method" not in cfg:
            schema_stale.append(meta)
            continue
        ws = cfg["working_space"]
        mtime = npy.stat().st_mtime
        if ws != "CIELAB":
            unaffected += 1
            continue
        if mtime >= cutoff:
            fresh += 1
            continue
        stale.append(meta)

    print(f"cache root         : {cache_dir}")
    print(f"cutoff             : {cutoff_str}")
    print(f"unaffected entries : {unaffected:>4d}  (working_space != CIELAB)")
    print(f"fresh CIELAB       : {fresh:>4d}  (built on or after cutoff)")
    print(f"STALE CIELAB       : {len(stale):>4d}  (built pre-fix; numerically wrong)")
    print(f"SCHEMA-STALE       : {len(schema_stale):>4d}  (pre-rename schema; no current LutConfig matches)")
    print()

    if stale:
        print("STALE CIELAB entries (build was pre-matrix-fix):")
        for meta in stale:
            npy = _npy_for(meta)
            cfg = json.loads(meta.read_text())
            mt = datetime.fromtimestamp(npy.stat().st_mtime).isoformat(timespec="seconds")
            size_mb = npy.stat().st_size / (1024 * 1024)
            key = meta.name[: -len(".meta.json")]
            sig = (f"shape={cfg['shape']:<8s} smoothing={cfg['smoothing']:<8s} "
                   f"metric={cfg['metric']:<10s} alpha_hat={cfg.get('metric_rgd_alpha_hat',0):>5.2f} "
                   f"interval={cfg['interval']:<3d} output_space={cfg['output_space']}")
            print(f"  {key}  {mt}  {size_mb:>6.1f} MB   {sig}")
        print()

    if schema_stale:
        print("SCHEMA-STALE entries (pre-rename; orphaned, no current key matches):")
        total_mb = 0.0
        for meta in schema_stale[:5]:
            npy = _npy_for(meta)
            mt = datetime.fromtimestamp(npy.stat().st_mtime).isoformat(timespec="seconds")
            size_mb = npy.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            key = meta.name[: -len(".meta.json")]
            print(f"  {key}  {mt}  {size_mb:>6.1f} MB")
        if len(schema_stale) > 5:
            print(f"  ... and {len(schema_stale) - 5} more (each ~192 MB)")
        # Sum without re-stating each
        all_mb = sum(_npy_for(m).stat().st_size / (1024 * 1024) for m in schema_stale)
        print(f"  total: {all_mb:.0f} MB across {len(schema_stale)} entries")
        print()

    nuke_targets = stale + (schema_stale if args.delete else [])
    if args.delete and nuke_targets:
        print(f"deleting {len(nuke_targets)} entries (stale + schema-stale)...")
        for meta in nuke_targets:
            key = meta.name[: -len(".meta.json")]
            npy = _npy_for(meta)
            max_idx = meta.parent / f"{key}_max_idx.csv"
            for q in (npy, meta, max_idx):
                if q.exists():
                    q.unlink()
        print("done")
    elif stale or schema_stale:
        print("run again with --delete to remove these.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
