"""Public clustering API with automatic CUDA/CPU backend selection.

The original scikit-learn implementation lives in ``clustering_cpu.py``.
When a CUDA-enabled PyTorch runtime is available, ``clustering_gpu.py`` is
selected. Existing imports from ``models.clustering`` remain compatible.
"""
from __future__ import annotations

import importlib
import warnings


def _cuda_available() -> bool:
    """Return True only when PyTorch can actually initialize CUDA."""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


CUDA_AVAILABLE = _cuda_available()
ACTIVE_BACKEND = "gpu" if CUDA_AVAILABLE else "cpu"

try:
    _backend = importlib.import_module(
        ".clustering_gpu" if CUDA_AVAILABLE else ".clustering_cpu",
        package=__package__,
    )
except Exception as exc:
    if CUDA_AVAILABLE:
        warnings.warn(
            f"Khong khoi tao duoc clustering_gpu ({exc}); chuyen sang clustering_cpu.",
            RuntimeWarning,
        )
    _backend = importlib.import_module(".clustering_cpu", package=__package__)
    ACTIVE_BACKEND = "cpu"

# Re-export the complete public API to preserve existing call sites.
for _name, _value in vars(_backend).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = [name for name in vars(_backend) if not name.startswith("_")] + [
    "CUDA_AVAILABLE", "ACTIVE_BACKEND"
]


def __getattr__(name: str):
    return getattr(_backend, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_backend)))
