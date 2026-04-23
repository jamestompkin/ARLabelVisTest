"""Scene loader for the `data/scenes/<name>/` layout.

Each scene directory holds:
  - ``video.mp4`` — the source video (equirectangular stereo in our case).
  - ``label_mask.png`` — binary PNG, foreground (>127) is the label region.
      Optional (may be ``null`` in the manifest).
  - ``manifest.json`` — metadata: fps, resolution, description, stereo layout.

Intended as the single source-of-truth so paper scripts and tests can enumerate
the available scenes without hardcoding paths. Add a new scene by dropping a
directory in this layout — no code changes needed for discovery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


SCENES_ROOT = Path(__file__).resolve().parents[1] / "data" / "scenes"


@dataclass(frozen=True)
class Scene:
    name: str
    dir: Path
    video: Path
    label_mask: Optional[Path]
    manifest: dict

    @property
    def fps(self) -> float:
        return float(self.manifest.get("fps", 30.0))

    @property
    def resolution(self) -> tuple[int, int]:
        return tuple(self.manifest.get("resolution", [0, 0]))

    @property
    def duration_s(self) -> float:
        return float(self.manifest.get("duration_s", 0.0))


def load_scene(name: str, root: Path = SCENES_ROOT) -> Scene:
    """Load the scene named ``name`` from ``data/scenes/<name>/``."""
    scene_dir = root / name
    manifest_path = scene_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    video = (scene_dir / manifest["video"]).resolve()
    if not video.exists():
        raise FileNotFoundError(f"video missing: {video}")

    mask_name = manifest.get("label_mask")
    label_mask: Optional[Path] = None
    if mask_name is not None:
        label_mask = scene_dir / mask_name
        if not label_mask.exists():
            raise FileNotFoundError(f"mask missing: {label_mask}")

    return Scene(name=name, dir=scene_dir, video=video,
                 label_mask=label_mask, manifest=manifest)


def iter_scenes(root: Path = SCENES_ROOT) -> Iterator[Scene]:
    """Yield every scene with a well-formed manifest under ``root``."""
    if not root.exists():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "manifest.json").exists():
            try:
                yield load_scene(child.name, root)
            except FileNotFoundError:
                continue
