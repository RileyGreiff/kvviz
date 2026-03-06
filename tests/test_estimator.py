"""Tests for the KV cache estimator."""

from kvviz.estimator import estimate_kv_cache, estimate_max_tokens
from kvviz.schema import DType, ModelConfig, RuntimeParams


def test_basic_estimate():
    """Known small example: 2 layers, 4 KV heads, head_dim=64, fp16, batch=1, seq=10."""
    config = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype=DType.fp16)
    params = RuntimeParams(batch=1, seq_len=10)
    result = estimate_kv_cache(config, params)
    # 2 layers * 2 (K+V) * 1 batch * 10 seq * 4 heads * 64 dim * 2 bytes = 20480
    assert result.total_bytes == 20480


def test_estimate_fp32():
    config = ModelConfig(num_layers=1, num_kv_heads=1, head_dim=128, dtype=DType.fp32)
    params = RuntimeParams(batch=2, seq_len=5)
    result = estimate_kv_cache(config, params)
    assert result.total_bytes == 1 * 2 * 2 * 5 * 1 * 128 * 4


def test_estimate_int8():
    config = ModelConfig(num_layers=4, num_kv_heads=2, head_dim=32, dtype=DType.int8)
    params = RuntimeParams(batch=1, seq_len=100)
    result = estimate_kv_cache(config, params)
    assert result.total_bytes == 4 * 2 * 1 * 100 * 2 * 32 * 1


def test_gqa_fallback():
    config = ModelConfig(num_layers=1, num_attn_heads=32, head_dim=64, dtype=DType.fp16)
    params = RuntimeParams(batch=1, seq_len=1)
    result = estimate_kv_cache(config, params)
    assert result.total_bytes == 1 * 2 * 1 * 1 * 32 * 64 * 2


def test_bytes_per_token():
    config = ModelConfig(num_layers=32, num_kv_heads=8, head_dim=128, dtype=DType.fp16)
    params = RuntimeParams(batch=1, seq_len=1)
    result = estimate_kv_cache(config, params)
    assert result.bytes_per_token == 32 * 2 * 8 * 128 * 2


def test_max_tokens():
    config = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype=DType.fp16)
    max_t = estimate_max_tokens(config, 1073741824, 0, batch=1)
    assert max_t == 1073741824 // (2 * 2 * 4 * 64 * 2)


def test_max_tokens_no_room():
    config = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64, dtype=DType.fp16)
    assert estimate_max_tokens(config, 1000, 2000, batch=1) == 0
