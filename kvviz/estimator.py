"""KV cache memory estimator (for planning without a GPU)."""

from __future__ import annotations

from dataclasses import dataclass

from kvviz.schema import ModelConfig, RuntimeParams
from kvviz.utils import get_bytes_per_element, format_bytes


@dataclass
class EstimateResult:
    """Result of a KV cache memory estimation."""

    total_bytes: int
    bytes_per_token: int
    bytes_per_generated_token: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    bytes_per_elem: int
    batch: int
    seq_len: int

    def as_dict(self) -> dict:
        return {
            "total_bytes": self.total_bytes,
            "total_human": format_bytes(self.total_bytes),
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_generated_token": self.bytes_per_generated_token,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "bytes_per_elem": self.bytes_per_elem,
            "batch": self.batch,
            "seq_len": self.seq_len,
        }


def estimate_kv_cache(config: ModelConfig, params: RuntimeParams) -> EstimateResult:
    """Compute KV cache memory usage.

    Formula:
        KV bytes = num_layers * 2 * batch * seq_len * num_kv_heads * head_dim * bytes_per_elem
    """
    bpe = get_bytes_per_element(config.dtype.value)
    kv_heads = config.effective_kv_heads()
    bytes_per_token = config.num_layers * 2 * kv_heads * config.head_dim * bpe
    total = bytes_per_token * params.batch * params.seq_len

    return EstimateResult(
        total_bytes=total,
        bytes_per_token=bytes_per_token,
        bytes_per_generated_token=bytes_per_token,
        num_layers=config.num_layers,
        num_kv_heads=kv_heads,
        head_dim=config.head_dim,
        dtype=config.dtype.value,
        bytes_per_elem=bpe,
        batch=params.batch,
        seq_len=params.seq_len,
    )


def estimate_max_tokens(
    config: ModelConfig,
    total_gpu_memory_bytes: int,
    reserved_bytes: int,
    batch: int = 1,
) -> int:
    """Estimate maximum sequence length before OOM."""
    available = total_gpu_memory_bytes - reserved_bytes
    if available <= 0:
        return 0
    bpe = get_bytes_per_element(config.dtype.value)
    kv_heads = config.effective_kv_heads()
    cost_per_token = config.num_layers * 2 * kv_heads * config.head_dim * bpe * batch
    if cost_per_token == 0:
        return 0
    return available // cost_per_token
