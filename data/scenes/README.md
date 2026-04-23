# Scenes

One directory per scene, discovered by `arlabelvis.scenes.iter_scenes()`.

## Layout

```
data/scenes/<scene_name>/
    label_mask.png    # binary mask: pixels > 127 are the label region
    manifest.json     # metadata (see below), including a path to the video
```

`manifest.json`'s `video` field may be a relative path (resolved from the scene
directory) — typical pattern for us is pointing it at an existing asset under
`external/unity/Assets/` rather than duplicating 30 MB of h264 into `data/`.

The scene *name* is its directory name. Keep it short and snake_cased.

## manifest.json

```json
{
  "name": "dubai_changing_bgcolors",
  "title": "Dubai skyline, changing background colors",
  "description": "One-paragraph summary.",
  "video": "video.mp4",
  "label_mask": "label_mask.png",
  "resolution": [3840, 1920],
  "stereo_layout": "top_bottom",
  "fps": 29.97,
  "duration_s": 15.22,
  "label_region": {
    "shape": "rect",
    "note": "Human-readable description of where the label sits."
  },
  "source": "external/unity/Assets/Dubai_Changing_BGColors.mp4",
  "mask_source": "provenance of the mask file"
}
```

`label_mask` may be `null` for scenes that don't yet have a mask checked in;
scripts that need a mask should skip such scenes with a clear message.

## Adding a new scene

1. `mkdir data/scenes/<name>`, drop in `video.mp4` + `label_mask.png`.
2. Write `manifest.json` following the schema above.
3. `iter_scenes()` picks it up automatically — no code changes required.

## Reproducing the paper's per-scene results

Paper-pipeline scene scripts (planned):
- `scripts/paper/tab_label_comparison.py` — per-scene flicker gradient stats.
- `scripts/paper/fig_first_frames.py` / `fig_second_frames.py` — frame sequences.

End-to-end test that a scene round-trips through the processor:
```
uv run python -m tests.test_dubai_scene
```
