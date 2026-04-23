"""One-shot cache migrations, idempotent.

Migrations applied (in order):

1. ``distance`` → ``metric`` field rename in every meta.json.
   Cache keys are unchanged (``LutConfig._HASH_FIELD_RENAMES`` serialises
   ``metric`` as ``distance`` for hashing), so the file names stay the same.
   This step just brings the human-readable metadata in line with the new
   field name.

2. Old ``metric/distance == "DeltaE76"`` entries get rewritten to the new
   canonical ``(metric="Euclidean", space="CIELAB", interp_space="CIELAB")``
   triple and renamed to the resulting hash.

3. Old ``space == "RGB"`` entries get rewritten to ``space="sRGB"`` and
   renamed to the new hash. "RGB" was ambiguous (sRGB? linear? gamma-
   encoded?); "sRGB" accurately names the 0-255 integer sRGB cube we
   actually operate on.

Safe to run multiple times — each step checks whether it needs to do anything.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.paper._lut_cache import CACHE_DIR, LutConfig


def _rename_field_in_meta(meta: dict) -> bool:
    """Rewrite ``distance`` → ``metric`` in-place. Returns True if it changed."""
    if "distance" in meta and "metric" not in meta:
        meta["metric"] = meta.pop("distance")
        return True
    return False


def _migrate_field_names() -> int:
    """Step 1: rewrite `distance` → `metric` in every meta.json."""
    changed = 0
    for meta_path in sorted(CACHE_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if _rename_field_in_meta(meta):
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            changed += 1
    if changed:
        print(f"[step 1] renamed `distance` -> `metric` in {changed} meta.json file(s)")
    return changed


def _migrate_deltae76_entries() -> int:
    """Step 2: rewrite legacy DeltaE76 entries to the canonical triple."""
    migrated = 0
    skipped = 0
    for meta_path in sorted(CACHE_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("metric") != "DeltaE76":
            continue

        old_key = meta_path.name[: -len(".meta.json")]
        new_cfg_kwargs = {k: v for k, v in meta.items()
                          if k in {"space", "smoothing", "sigma",
                                   "alpha_hat", "interval", "voxel_dim"}}
        new_cfg = LutConfig(metric="Euclidean", interp_space="CIELAB",
                            **new_cfg_kwargs)
        new_key = new_cfg.key()

        if new_key == old_key:
            print(f"[skip] {old_key}: new hash matches (no rename needed)")
            continue

        # Materialise siblings before renaming so Windows glob isn't confused
        # by in-flight renames.
        siblings = sorted(CACHE_DIR.glob(f"{old_key}*"))
        for sibling in siblings:
            target = CACHE_DIR / sibling.name.replace(old_key, new_key, 1)
            if target.exists():
                print(f"[warn] target {target.name} already exists, skipping")
                skipped += 1
                continue
            sibling.rename(target)

        # Rewrite the meta to reflect the canonical triple.
        new_meta_path = CACHE_DIR / f"{new_key}.meta.json"
        new_meta = json.loads(new_meta_path.read_text())
        new_meta["metric"] = "Euclidean"
        new_meta["interp_space"] = "CIELAB"
        new_meta_path.write_text(json.dumps(new_meta, indent=2, sort_keys=True))
        print(f"[migrate] {old_key} -> {new_key}")
        migrated += 1

    if migrated or skipped:
        print(f"[step 2] {migrated} DeltaE76 entries migrated, {skipped} skipped")
    return migrated


def _migrate_rgb_space() -> int:
    """Step 3: rewrite legacy ``space="RGB"`` entries to ``space="sRGB"``."""
    migrated = 0
    for meta_path in sorted(CACHE_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("space") != "RGB":
            continue

        old_key = meta_path.name[: -len(".meta.json")]
        new_kwargs = {k: v for k, v in meta.items()
                      if k in {"smoothing", "sigma", "alpha_hat",
                               "interval", "voxel_dim", "metric",
                               "interp_space"}}
        new_cfg = LutConfig(space="sRGB", **new_kwargs)
        new_key = new_cfg.key()

        if new_key == old_key:
            continue

        siblings = sorted(CACHE_DIR.glob(f"{old_key}*"))
        for sibling in siblings:
            target = CACHE_DIR / sibling.name.replace(old_key, new_key, 1)
            if target.exists():
                continue
            sibling.rename(target)

        new_meta_path = CACHE_DIR / f"{new_key}.meta.json"
        new_meta = json.loads(new_meta_path.read_text())
        new_meta["space"] = "sRGB"
        new_meta_path.write_text(json.dumps(new_meta, indent=2, sort_keys=True))
        print(f"[migrate] {old_key} -> {new_key}  (space: RGB -> sRGB)")
        migrated += 1

    if migrated:
        print(f"[step 3] {migrated} RGB-space entries migrated")
    return migrated


def main() -> int:
    _migrate_field_names()
    _migrate_deltae76_entries()
    _migrate_rgb_space()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
