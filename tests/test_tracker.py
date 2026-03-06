"""Tests for the KV cache tracker (unit tests that don't require a GPU)."""

import pytest

from kvviz.schema import DType, EventType, ModelConfig, Trace
from kvviz.tracker import _measure_cache, save_trace, load_trace


def test_measure_cache_none():
    total, layers = _measure_cache(None)
    assert total == 0
    assert layers == []


def test_measure_cache_tuple_format():
    """Test measuring KV cache from legacy tuple-of-tuples format."""
    pytest.importorskip("torch")
    import torch

    # Simulate 2 layers, batch=1, 4 heads, seq_len=10, head_dim=64, fp16
    k = torch.zeros(1, 4, 10, 64, dtype=torch.float16)
    v = torch.zeros(1, 4, 10, 64, dtype=torch.float16)
    past = ((k, v), (k.clone(), v.clone()))

    total, layers = _measure_cache(past)
    expected_per_tensor = 1 * 4 * 10 * 64 * 2  # 5120 bytes
    assert total == expected_per_tensor * 4  # 2 layers * (K + V)
    assert len(layers) == 2
    assert layers[0].key_bytes == expected_per_tensor
    assert layers[0].value_bytes == expected_per_tensor
    assert layers[0].seq_len == 10


def test_save_load_roundtrip(tmp_path):
    trace = Trace(
        model_name="test",
        config=ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype=DType.fp16),
        device="cpu",
        events=[],
    )
    path = tmp_path / "trace.json"
    save_trace(trace, path)
    loaded = load_trace(path)
    assert loaded.model_name == "test"
    assert loaded.config.num_layers == 2
