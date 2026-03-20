"""Data models for fragmentation simulation traces."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AllocatorMode(str, Enum):
    paged = "paged"
    contiguous = "contiguous"


class FreePolicy(str, Enum):
    immediate = "immediate"
    end_of_request = "end_of_request"


class FragEventType(str, Enum):
    request_arrive = "request_arrive"
    token_generate = "token_generate"
    request_finish = "request_finish"


class TrafficRequest(BaseModel):
    """A single request in a synthetic traffic trace."""

    request_id: str
    arrival_step: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., gt=0)
    gen_tokens: int = Field(..., gt=0)


class TrafficTrace(BaseModel):
    """Synthetic multi-request traffic trace."""

    requests: list[TrafficRequest] = Field(default_factory=list)
    total_steps: int = Field(0, ge=0)


class RequestAllocationState(BaseModel):
    """Per-request allocation snapshot."""

    request_id: str
    live_tokens: int = 0
    allocated_blocks: int = 0
    allocated_capacity_tokens: int = 0
    used_tokens: int = 0
    tail_waste_tokens: int = 0
    packing_efficiency: float = 1.0


class GlobalMetrics(BaseModel):
    """Global block-level metrics at a simulation snapshot."""

    total_blocks: int = 0
    free_blocks: int = 0
    used_blocks: int = 0
    largest_free_run: int = 0
    fragmentation_ratio: float = 0.0
    blocks_allocated_this_step: int = 0
    blocks_freed_this_step: int = 0
    reuse_count: int = 0


class FragEvent(BaseModel):
    """A single fragmentation simulation event."""

    event_idx: int = Field(..., ge=0)
    event_type: FragEventType
    request_id: str
    step: int = Field(..., ge=0)
    requests: list[RequestAllocationState] = Field(default_factory=list)
    global_metrics: GlobalMetrics = Field(default_factory=GlobalMetrics)
    block_map: Optional[list[Optional[str]]] = Field(
        None, description="Block ownership map: request_id or None for free"
    )


class FragSimConfig(BaseModel):
    """Configuration for a fragmentation simulation run."""

    block_size_tokens: int = Field(16, gt=0)
    allocator: AllocatorMode = AllocatorMode.paged
    max_blocks: Optional[int] = None
    cache_window_tokens: Optional[int] = None
    free_policy: FreePolicy = FreePolicy.immediate


class FragTrace(BaseModel):
    """Complete fragmentation simulation trace."""

    config: FragSimConfig = Field(default_factory=FragSimConfig)
    events: list[FragEvent] = Field(default_factory=list)
    source: str = "unknown"


# ---------------------------------------------------------------------------
# Live GPU observation models
# ---------------------------------------------------------------------------


class CudaMemoryStats(BaseModel):
    """Snapshot of PyTorch CUDA allocator statistics."""

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    active_bytes: int = 0
    inactive_split_bytes: int = 0
    fragmentation_ratio: float = 0.0
    num_alloc_retries: int = 0


class FragSnapshot(BaseModel):
    """Lightweight snapshot for live streaming (vs. full FragEvent lifecycle)."""

    timestamp_ms: float
    total_blocks: int = 0
    free_blocks: int = 0
    used_blocks: int = 0
    largest_free_run: int = 0
    fragmentation_ratio: float = 0.0
    packing_efficiency: float = 1.0
    block_map: Optional[list[Optional[str]]] = None
    requests: list[RequestAllocationState] = Field(default_factory=list)
    cuda_stats: Optional[CudaMemoryStats] = None
    reuse_count: int = 0
    step: int = 0
