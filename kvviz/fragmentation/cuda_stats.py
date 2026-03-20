"""CUDA memory statistics capture for live fragmentation monitoring."""

from __future__ import annotations

import logging

from kvviz.fragmentation.schema import CudaMemoryStats

logger = logging.getLogger("kvviz")


def capture_cuda_stats(device: int = 0) -> CudaMemoryStats | None:
    """Capture current CUDA allocator statistics.

    Returns ``None`` when ``torch.cuda`` is unavailable (CPU-only
    environment).  The call is O(1) and safe for high-frequency polling.
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        stats = torch.cuda.memory_stats(device)
    except RuntimeError:
        return None

    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    active = stats.get("active_bytes.all.current", 0)
    inactive_split = stats.get("inactive_split_bytes.all.current", 0)
    retries = stats.get("num_alloc_retries", 0)

    frag = (reserved - active) / reserved if reserved > 0 else 0.0

    return CudaMemoryStats(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        active_bytes=active,
        inactive_split_bytes=inactive_split,
        fragmentation_ratio=frag,
        num_alloc_retries=retries,
    )
