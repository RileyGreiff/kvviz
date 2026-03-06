"""Shared utilities: dtype mapping, logging, formatting, GPU helpers."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "float32": 4,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "int8": 1,
}

VALID_DTYPES = ["fp32", "fp16", "bf16", "fp8", "int8"]


def get_bytes_per_element(dtype: str) -> int:
    """Return bytes per element for the given dtype string."""
    dtype = dtype.lower().replace("torch.", "")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype '{dtype}'. Valid: {VALID_DTYPES}")
    return DTYPE_BYTES[dtype]


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the kvviz logger."""
    logger = logging.getLogger("kvviz")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def format_bytes(n: int | float) -> str:
    """Human-readable byte string."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory state."""
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int

    @staticmethod
    def capture(device: int = 0) -> "GPUMemorySnapshot":
        """Capture current GPU memory state via torch.cuda."""
        try:
            import torch
            if not torch.cuda.is_available():
                return GPUMemorySnapshot(0, 0, 0, 0)
            return GPUMemorySnapshot(
                total_bytes=torch.cuda.get_device_properties(device).total_mem,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                free_bytes=torch.cuda.get_device_properties(device).total_mem - torch.cuda.memory_allocated(device),
            )
        except ImportError:
            return GPUMemorySnapshot(0, 0, 0, 0)
