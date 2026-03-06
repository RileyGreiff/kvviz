"""Synthetic trace generator for testing without a GPU."""

from __future__ import annotations

import random

from kvviz.estimator import estimate_kv_cache
from kvviz.schema import (
    DType,
    EventType,
    LayerKVSnapshot,
    ModelConfig,
    RuntimeParams,
    Trace,
    TraceEvent,
)
from kvviz.utils import get_bytes_per_element


def generate_trace(
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: DType = DType.fp16,
    prompt_tokens: int = 128,
    max_new_tokens: int = 64,
    seed: int | None = None,
    max_cache_tokens: int | None = None,
    on_event: Any = None,
) -> Trace:
    """Generate a synthetic trace that mimics real monitored inference.

    Produces events matching the live tracker format:
    prefill_start -> prefill_end -> decode_step * N -> generation_end
    """
    if seed is not None:
        random.seed(seed)

    config = ModelConfig(
        model_name="synthetic",
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    bpe = get_bytes_per_element(dtype.value)
    events: list[TraceEvent] = []
    ts = 0.0
    peak_kv = 0

    def _make_layer_snapshots(seq_len: int) -> tuple[list[LayerKVSnapshot], int]:
        """Build per-layer snapshots and return (layers, total_bytes)."""
        layers = []
        total = 0
        for i in range(num_layers):
            # Each layer: key and value each have shape (batch, kv_heads, seq_len, head_dim)
            per_tensor = seq_len * num_kv_heads * head_dim * bpe
            layers.append(LayerKVSnapshot(
                layer_idx=i, key_bytes=per_tensor, value_bytes=per_tensor, seq_len=seq_len,
            ))
            total += per_tensor * 2
        return layers, total

    def _emit(event: TraceEvent) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    # prefill_start
    _emit(TraceEvent(
        timestamp_ms=ts, event_type=EventType.prefill_start, step=0,
        total_tokens=prompt_tokens, prompt_tokens=prompt_tokens, generated_tokens=0,
        kv_bytes_total=0, kv_bytes_peak=0, layers=[],
    ))
    ts += random.uniform(0.5, 2.0)

    # prefill_end — apply eviction if window is smaller than prompt
    cache_seq = prompt_tokens
    if max_cache_tokens is not None and cache_seq > max_cache_tokens:
        cache_seq = max_cache_tokens
    layers, kv_total = _make_layer_snapshots(cache_seq)
    peak_kv = kv_total
    prefill_ms = random.uniform(10.0, 100.0) * (prompt_tokens / 128)
    _emit(TraceEvent(
        timestamp_ms=ts, event_type=EventType.prefill_end, step=0,
        total_tokens=prompt_tokens, prompt_tokens=prompt_tokens, generated_tokens=0,
        kv_bytes_total=kv_total, kv_bytes_peak=peak_kv, layers=layers,
        step_latency_ms=prefill_ms,
    ))
    ts += prefill_ms

    # decode steps
    for step in range(1, max_new_tokens + 1):
        current_seq = prompt_tokens + step
        # Apply sliding window eviction
        cache_seq = current_seq
        if max_cache_tokens is not None and cache_seq > max_cache_tokens:
            cache_seq = max_cache_tokens
        layers, kv_total = _make_layer_snapshots(cache_seq)
        peak_kv = max(peak_kv, kv_total)

        # Simulate latency based on cache size (not total tokens)
        base_latency = random.uniform(8.0, 15.0)
        scaling = 1.0 + 0.3 * (cache_seq / (prompt_tokens + max_new_tokens))
        step_ms = base_latency * scaling

        _emit(TraceEvent(
            timestamp_ms=ts, event_type=EventType.decode_step, step=step,
            total_tokens=current_seq, prompt_tokens=prompt_tokens,
            generated_tokens=step,
            kv_bytes_total=kv_total, kv_bytes_peak=peak_kv, layers=layers,
            step_latency_ms=step_ms,
        ))
        ts += step_ms

    # generation_end
    _emit(TraceEvent(
        timestamp_ms=ts, event_type=EventType.generation_end, step=max_new_tokens,
        total_tokens=prompt_tokens + max_new_tokens, prompt_tokens=prompt_tokens,
        generated_tokens=max_new_tokens,
        kv_bytes_total=kv_total, kv_bytes_peak=peak_kv, layers=layers,
    ))

    return Trace(
        model_name="synthetic",
        config=config,
        device="cpu",
        events=events,
    )
