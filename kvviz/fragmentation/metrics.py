"""Metric computation for fragmentation simulation."""

from __future__ import annotations

from kvviz.fragmentation.allocator import BlockAllocator
from kvviz.fragmentation.schema import GlobalMetrics, RequestAllocationState


def compute_request_metrics(
    request_id: str,
    live_tokens: int,
    allocator: BlockAllocator,
) -> RequestAllocationState:
    """Compute per-request allocation metrics."""
    allocated_blocks = allocator.get_request_blocks(request_id)
    allocated_capacity = allocated_blocks * allocator.block_size_tokens
    used = min(live_tokens, allocated_capacity)
    tail_waste = allocated_capacity - used
    efficiency = used / allocated_capacity if allocated_capacity > 0 else 1.0
    return RequestAllocationState(
        request_id=request_id,
        live_tokens=live_tokens,
        allocated_blocks=allocated_blocks,
        allocated_capacity_tokens=allocated_capacity,
        used_tokens=used,
        tail_waste_tokens=tail_waste,
        packing_efficiency=efficiency,
    )


def compute_global_metrics(allocator: BlockAllocator) -> GlobalMetrics:
    """Compute global block-level metrics from allocator state."""
    total = allocator.total_blocks
    free = allocator.free_blocks
    used = total - free
    lfr = allocator.largest_free_run()
    frag = 1.0 - (lfr / free) if free > 0 else 0.0
    return GlobalMetrics(
        total_blocks=total,
        free_blocks=free,
        used_blocks=used,
        largest_free_run=lfr,
        fragmentation_ratio=frag,
        blocks_allocated_this_step=allocator.blocks_allocated_this_step,
        blocks_freed_this_step=allocator.blocks_freed_this_step,
        reuse_count=allocator.reuse_count,
    )
