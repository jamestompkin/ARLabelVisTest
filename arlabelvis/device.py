"""GPU device auto-detection.

Returns ``"cuda"`` when an NVIDIA GPU with CUDA support is available,
``"cpu"`` otherwise.  Logs the device name and VRAM once on first call.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)
_logged = False


def default_device() -> str:
    """Return the best available torch device string."""
    import torch

    global _logged
    if torch.cuda.is_available():
        if not _logged:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            _log.info("Using CUDA device: %s (%.1f GB VRAM)", name, vram)
            _logged = True
        return "cuda"
    if not _logged:
        _log.info("No CUDA device found; using CPU")
        _logged = True
    return "cpu"
