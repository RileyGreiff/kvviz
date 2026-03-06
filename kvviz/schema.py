"""Pydantic models for model config, trace events, and snapshots."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DType(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"
    fp8 = "fp8"
    int8 = "int8"


class ModelConfig(BaseModel):
    """Transformer model configuration relevant to KV cache sizing."""

    model_name: str = "unknown"
    num_layers: int = Field(..., gt=0)
    num_attn_heads: Optional[int] = Field(None, gt=0)
    num_kv_heads: Optional[int] = Field(None, gt=0)
    head_dim: int = Field(..., gt=0)
    dtype: DType = Field(DType.fp16)

    def effective_kv_heads(self) -> int:
        if self.num_kv_heads is not None:
            return self.num_kv_heads
        if self.num_attn_heads is not None:
            return self.num_attn_heads
        raise ValueError("Either num_kv_heads or num_attn_heads must be provided")


class RuntimeParams(BaseModel):
    """Runtime parameters for KV cache estimation."""

    batch: int = Field(1, gt=0)
    seq_len: int = Field(512, gt=0)


# --- Live monitoring event types ---


class EventType(str, Enum):
    prefill_start = "prefill_start"
    prefill_end = "prefill_end"
    decode_step = "decode_step"
    generation_end = "generation_end"


class LayerKVSnapshot(BaseModel):
    """KV cache size for a single layer at a point in time."""

    layer_idx: int = Field(..., ge=0)
    key_bytes: int = Field(0, ge=0)
    value_bytes: int = Field(0, ge=0)
    seq_len: int = Field(0, ge=0)

    @property
    def total_bytes(self) -> int:
        return self.key_bytes + self.value_bytes


class TraceEvent(BaseModel):
    """A single KV cache monitoring event with per-layer detail."""

    timestamp_ms: float = Field(..., ge=0)
    event_type: EventType
    step: int = Field(0, ge=0, description="Decode step number (0 = prefill)")
    total_tokens: int = Field(0, ge=0, description="Current total sequence length")
    prompt_tokens: int = Field(0, ge=0)
    generated_tokens: int = Field(0, ge=0)
    kv_bytes_total: int = Field(0, ge=0, description="Total KV cache bytes across all layers")
    kv_bytes_peak: int = Field(0, ge=0)
    layers: list[LayerKVSnapshot] = Field(default_factory=list)
    step_latency_ms: Optional[float] = Field(None, ge=0, description="Time for this decode step")
    gpu_allocated_bytes: Optional[int] = Field(None, ge=0)
    gpu_reserved_bytes: Optional[int] = Field(None, ge=0)


class Trace(BaseModel):
    """A complete trace from a monitored generation run."""

    model_name: str = "unknown"
    config: Optional[ModelConfig] = None
    device: str = "unknown"
    events: list[TraceEvent] = Field(default_factory=list)

    @property
    def peak_kv_bytes(self) -> int:
        if not self.events:
            return 0
        return max(e.kv_bytes_peak for e in self.events)

    @property
    def total_generated(self) -> int:
        if not self.events:
            return 0
        return max(e.generated_tokens for e in self.events)
