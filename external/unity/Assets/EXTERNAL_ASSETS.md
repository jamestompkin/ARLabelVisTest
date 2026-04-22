# External assets used by this Unity project

Third-party assets live in `external/` next to the Unity project itself (so this file's parent is `external/unity/Assets/`, and the Fantasy Forest asset is at `external/unity_fantasy_forest/` — two levels up). Restore the assets into this `Assets/` directory before opening Unity.

## Assets to restore

| External location (relative to this file's parent) | Unity Assets path |
| --- | --- |
| `../../unity_fantasy_forest/` | `Fantasy Forest Environment Free Sample/` |
| `../../unity_fantasy_forest.meta` | `Fantasy Forest Environment Free Sample.meta` |

## Restore via symlink (preferred)

Windows (developer mode or admin command prompt), from `external/unity/Assets/`:

```
mklink /D "Fantasy Forest Environment Free Sample" "..\..\unity_fantasy_forest"
mklink    "Fantasy Forest Environment Free Sample.meta" "..\..\unity_fantasy_forest.meta"
```

macOS / Linux, from `external/unity/Assets/`:

```sh
ln -s ../../unity_fantasy_forest "Fantasy Forest Environment Free Sample"
ln -s ../../unity_fantasy_forest.meta "Fantasy Forest Environment Free Sample.meta"
```

## Restore via copy (fallback)

If symlinks are unavailable, copy the directory instead. This duplicates files; the copy under `Assets/` is untracked by git (see `external/unity/.gitignore`).
