"""Public evaluation API; selects GPU backend only when CUDA is available.

The CPU implementation is kept in evaluation_cpu.py. The CUDA implementation
is kept in evaluation_gpu.py. Existing imports from models.evaluation remain
valid and do not need to change.
"""
from __future__ import annotations
import importlib
import warnings


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

CUDA_AVAILABLE = _cuda_available()
ACTIVE_BACKEND = "gpu" if CUDA_AVAILABLE else "cpu"
try:
    _backend = importlib.import_module(
        ".evaluation_gpu" if CUDA_AVAILABLE else ".evaluation_cpu", package=__package__
    )
except Exception as exc:
    if CUDA_AVAILABLE:
        warnings.warn(
            f"Không khởi tạo được evaluation_gpu ({exc}); chuyển sang evaluation_cpu.",
            RuntimeWarning,
        )
    _backend = importlib.import_module(".evaluation_cpu", package=__package__)
    ACTIVE_BACKEND = "cpu"

# Re-export every original module attribute for backward compatibility.
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
