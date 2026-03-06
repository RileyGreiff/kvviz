"""Tests for synthetic trace generation."""

from kvviz.schema import EventType
from kvviz.synth import generate_trace


def test_event_ordering():
    """Events follow: prefill_start -> prefill_end -> decode_steps -> generation_end."""
    trace = generate_trace(seed=123)
    types = [e.event_type for e in trace.events]
    assert types[0] == EventType.prefill_start
    assert types[1] == EventType.prefill_end
    assert types[-1] == EventType.generation_end
    for t in types[2:-1]:
        assert t == EventType.decode_step


def test_monotonic_timestamps():
    trace = generate_trace(seed=42)
    timestamps = [e.timestamp_ms for e in trace.events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1]


def test_kv_growth():
    """KV bytes should grow during decode steps."""
    trace = generate_trace(seed=99)
    decode_events = [e for e in trace.events if e.event_type == EventType.decode_step]
    assert len(decode_events) > 1
    for i in range(1, len(decode_events)):
        assert decode_events[i].kv_bytes_total >= decode_events[i - 1].kv_bytes_total


def test_has_per_layer_data():
    trace = generate_trace(num_layers=4, seed=1)
    decode_events = [e for e in trace.events if e.event_type == EventType.decode_step]
    assert len(decode_events) > 0
    assert len(decode_events[0].layers) == 4
    for snap in decode_events[0].layers:
        assert snap.key_bytes > 0
        assert snap.value_bytes > 0


def test_has_config():
    trace = generate_trace(num_layers=16, seed=1)
    assert trace.config is not None
    assert trace.config.num_layers == 16


def test_step_latencies():
    trace = generate_trace(seed=10)
    decode_events = [e for e in trace.events if e.event_type == EventType.decode_step]
    for e in decode_events:
        assert e.step_latency_ms is not None
        assert e.step_latency_ms > 0


def test_token_counts():
    trace = generate_trace(prompt_tokens=100, max_new_tokens=30, seed=5)
    end = [e for e in trace.events if e.event_type == EventType.generation_end]
    assert len(end) == 1
    assert end[0].prompt_tokens == 100
    assert end[0].generated_tokens == 30
    assert end[0].total_tokens == 130
