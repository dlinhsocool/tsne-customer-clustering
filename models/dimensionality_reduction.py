"""Public dimensionality-reduction API with automatic CUDA/CPU backend selection.

CPU base implementation: dimensionality_reduction_cpu.py
Hand-written CUDA implementation: dimensionality_reduction_gpu.py
Existing imports from models.dimensionality_reduction remain compatible.
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
        ".dimensionality_reduction_gpu" if CUDA_AVAILABLE else ".dimensionality_reduction_cpu",
        package=__package__,
    )
except Exception as exc:
    if CUDA_AVAILABLE:
        warnings.warn(
            f"Không khởi tạo được dimensionality_reduction_gpu ({exc}); chuyển sang CPU. Lý do: {exc}",
            RuntimeWarning,
        )
    _backend = importlib.import_module(".dimensionality_reduction_cpu", package=__package__)
    ACTIVE_BACKEND = "cpu"

for _name, _value in vars(_backend).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = [name for name in vars(_backend) if not name.startswith("_")] + ["CUDA_AVAILABLE", "ACTIVE_BACKEND"]

def __getattr__(name: str):
    return getattr(_backend, name)

def __dir__():
    return sorted(set(globals()) | set(dir(_backend)))
