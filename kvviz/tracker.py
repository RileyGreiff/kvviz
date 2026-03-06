"""Live KV cache tracker that hooks into PyTorch models during inference.

Supports:
- HuggingFace models using DynamicCache / transformers cache objects
- Any PyTorch model by hooking attention layer outputs
- Manual instrumentation via the record() API

Usage with HuggingFace generate():

    from kvviz.tracker import KVCacheTracker

    tracker = KVCacheTracker.from_model(model, tokenizer)
    trace = tracker.generate(prompt, max_new_tokens=100)
    trace.save("trace.json")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from kvviz.schema import (
    DType,
    EventType,
    LayerKVSnapshot,
    ModelConfig,
    Trace,
    TraceEvent,
)

logger = logging.getLogger("kvviz")


def _detect_dtype(dtype_obj: Any) -> DType:
    """Map a torch dtype to our DType enum."""
    dtype_str = str(dtype_obj).replace("torch.", "")
    mapping = {
        "float32": DType.fp32,
        "float16": DType.fp16,
        "bfloat16": DType.bf16,
        "int8": DType.int8,
        "float8_e4m3fn": DType.fp8,
        "float8_e5m2": DType.fp8,
    }
    return mapping.get(dtype_str, DType.fp16)


def _extract_model_config(model: Any) -> ModelConfig:
    """Extract ModelConfig from a HuggingFace model."""
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", None)
    num_attn_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_attn_heads)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden = getattr(config, "hidden_size", None)
        if hidden and num_attn_heads:
            head_dim = hidden // num_attn_heads

    # Detect dtype from model parameters
    dtype = DType.fp16
    try:
        first_param = next(model.parameters())
        dtype = _detect_dtype(first_param.dtype)
    except StopIteration:
        pass

    model_name = getattr(config, "_name_or_path", "unknown")

    return ModelConfig(
        model_name=model_name,
        num_layers=num_layers or 1,
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim or 64,
        dtype=dtype,
    )


def _measure_tensor_pair(k: Any, v: Any, layer_idx: int) -> LayerKVSnapshot | None:
    """Measure a single (key, value) tensor pair."""
    import torch
    if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
        kb = k.nelement() * k.element_size()
        vb = v.nelement() * v.element_size()
        seq = k.shape[-2] if k.dim() >= 2 else 0
        return LayerKVSnapshot(layer_idx=layer_idx, key_bytes=kb, value_bytes=vb, seq_len=seq)
    return None


def _measure_cache(past_key_values: Any) -> tuple[int, list[LayerKVSnapshot]]:
    """Measure KV cache size from a HuggingFace cache object.

    Supports:
    - DynamicCache with .layers list (transformers >= 5.x) — each layer has .keys/.values
    - DynamicCache with .key_cache/.value_cache lists (transformers 4.36–4.x)
    - Legacy tuple-of-tuples format ((key, value), ...)

    Returns (total_bytes, list_of_layer_snapshots).
    """
    layers: list[LayerKVSnapshot] = []
    total_bytes = 0

    if past_key_values is None:
        return 0, []

    # transformers >= 5.x DynamicCache: has .layers list of DynamicLayer objects
    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, list):
        for i, layer in enumerate(past_key_values.layers):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            snap = _measure_tensor_pair(k, v, i)
            if snap:
                layers.append(snap)
                total_bytes += snap.key_bytes + snap.value_bytes
        if layers:
            return total_bytes, layers

    # transformers 4.36+ DynamicCache: has .key_cache and .value_cache lists
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            snap = _measure_tensor_pair(k, v, i)
            if snap:
                layers.append(snap)
                total_bytes += snap.key_bytes + snap.value_bytes
        if layers:
            return total_bytes, layers

    # Legacy tuple-of-tuples: ((key, value), (key, value), ...)
    if isinstance(past_key_values, (tuple, list)):
        for i, layer_kv in enumerate(past_key_values):
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                snap = _measure_tensor_pair(layer_kv[0], layer_kv[1], i)
                if snap:
                    layers.append(snap)
                    total_bytes += snap.key_bytes + snap.value_bytes
        if layers:
            return total_bytes, layers

    logger.warning("Unknown cache type: %s. Cannot measure KV cache.", type(past_key_values))
    return 0, []


def _evict_cache(past_key_values: Any, max_tokens: int) -> Any:
    """Sliding-window eviction: trim KV cache to keep only the last max_tokens.

    Returns the trimmed cache object (same type as input).
    """
    import torch

    # DynamicCache with .key_cache/.value_cache lists
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i in range(len(past_key_values.key_cache)):
            seq_dim = -2
            seq_len = past_key_values.key_cache[i].shape[seq_dim]
            if seq_len > max_tokens:
                past_key_values.key_cache[i] = past_key_values.key_cache[i][..., -max_tokens:, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][..., -max_tokens:, :]
        return past_key_values

    # DynamicCache with .layers (transformers >= 5.x)
    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, list):
        for layer in past_key_values.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, torch.Tensor) and k.shape[-2] > max_tokens:
                layer.keys = k[..., -max_tokens:, :]
                layer.values = v[..., -max_tokens:, :]
        return past_key_values

    # Legacy tuple-of-tuples
    if isinstance(past_key_values, (tuple, list)):
        new_cache = []
        for layer_kv in past_key_values:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                k, v = layer_kv[0], layer_kv[1]
                if isinstance(k, torch.Tensor) and k.shape[-2] > max_tokens:
                    k = k[..., -max_tokens:, :]
                    v = v[..., -max_tokens:, :]
                new_cache.append((k, v) + tuple(layer_kv[2:]))
            else:
                new_cache.append(layer_kv)
        return type(past_key_values)(new_cache)

    return past_key_values


class KVCacheTracker:
    """Live KV cache monitor for transformer inference.

    Attaches to a HuggingFace model and records KV cache size, per-layer
    breakdown, decode latency, and GPU memory at each generation step.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_config: Optional[ModelConfig] = None,
        device: Optional[str] = None,
        max_cache_tokens: Optional[int] = None,
        on_event: Optional[Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config or _extract_model_config(model)
        self._events: list[TraceEvent] = []
        self._peak_kv: int = 0
        self._start_time_ms: float = 0.0
        self.max_cache_tokens: Optional[int] = max_cache_tokens
        self.on_event = on_event

        if device is None:
            try:
                self.device = str(next(model.parameters()).device)
            except StopIteration:
                self.device = "cpu"
        else:
            self.device = device

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        max_cache_tokens: Optional[int] = None,
        on_event: Optional[Any] = None,
    ) -> "KVCacheTracker":
        """Create a tracker from a HuggingFace model + tokenizer."""
        return cls(model=model, tokenizer=tokenizer, device=device,
                   max_cache_tokens=max_cache_tokens, on_event=on_event)

    def _now_ms(self) -> float:
        return (time.perf_counter() * 1000) - self._start_time_ms

    def _gpu_snapshot(self) -> tuple[Optional[int], Optional[int]]:
        """Return (allocated_bytes, reserved_bytes) or (None, None)."""
        try:
            import torch
            if torch.cuda.is_available() and "cuda" in self.device:
                dev = int(self.device.split(":")[-1]) if ":" in self.device else 0
                return (
                    torch.cuda.memory_allocated(dev),
                    torch.cuda.memory_reserved(dev),
                )
        except Exception:
            pass
        return None, None

    def _record_event(
        self,
        event_type: EventType,
        step: int,
        prompt_tokens: int,
        generated_tokens: int,
        past_key_values: Any,
        step_latency_ms: Optional[float] = None,
    ) -> None:
        """Record a single trace event by measuring the cache object."""
        ts = self._now_ms()
        kv_total, layer_snapshots = _measure_cache(past_key_values)
        self._peak_kv = max(self._peak_kv, kv_total)
        gpu_alloc, gpu_reserved = self._gpu_snapshot()

        event = TraceEvent(
            timestamp_ms=ts,
            event_type=event_type,
            step=step,
            total_tokens=prompt_tokens + generated_tokens,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            kv_bytes_total=kv_total,
            kv_bytes_peak=self._peak_kv,
            layers=layer_snapshots,
            step_latency_ms=step_latency_ms,
            gpu_allocated_bytes=gpu_alloc,
            gpu_reserved_bytes=gpu_reserved,
        )
        self._events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generate_kwargs: Any,
    ) -> Trace:
        """Run model.generate() with live KV cache monitoring.

        This wraps HuggingFace's generate() and captures KV cache state
        at each decode step using a streamer-like callback.

        Returns a Trace with all recorded events.
        """
        import torch

        self._events = []
        self._peak_kv = 0
        self._start_time_ms = time.perf_counter() * 1000

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        prompt_len = input_ids.shape[1]
        logger.info("Starting monitored generation: prompt_tokens=%d, max_new_tokens=%d", prompt_len, max_new_tokens)

        # --- Prefill ---
        prefill_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        prefill_ms = (time.perf_counter() - prefill_start) * 1000
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        self._record_event(
            EventType.prefill_start, step=0, prompt_tokens=prompt_len,
            generated_tokens=0, past_key_values=None,
        )
        self._record_event(
            EventType.prefill_end, step=0, prompt_tokens=prompt_len,
            generated_tokens=0, past_key_values=past_key_values,
            step_latency_ms=prefill_ms,
        )

        logger.info("Prefill done in %.1fms, KV cache: %d bytes", prefill_ms, self._peak_kv)

        # --- Decode loop ---
        generated_ids: list[int] = []
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_set = set(eos_token_id)
        elif eos_token_id is not None:
            eos_set = {eos_token_id}
        else:
            eos_set = set()

        for step in range(1, max_new_tokens + 1):
            token_id = next_token.item()
            generated_ids.append(token_id)

            if token_id in eos_set:
                self._record_event(
                    EventType.generation_end, step=step, prompt_tokens=prompt_len,
                    generated_tokens=len(generated_ids), past_key_values=past_key_values,
                )
                logger.info("EOS at step %d", step)
                break

            step_start = time.perf_counter()

            # Build attention mask for this step
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.model.device, dtype=attention_mask.dtype),
                ], dim=1)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            step_ms = (time.perf_counter() - step_start) * 1000
            past_key_values = outputs.past_key_values

            if self.max_cache_tokens is not None:
                past_key_values = _evict_cache(past_key_values, self.max_cache_tokens)
                if attention_mask is not None and attention_mask.shape[1] > self.max_cache_tokens + 1:
                    attention_mask = attention_mask[:, -(self.max_cache_tokens + 1):]

            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            self._record_event(
                EventType.decode_step, step=step, prompt_tokens=prompt_len,
                generated_tokens=len(generated_ids), past_key_values=past_key_values,
                step_latency_ms=step_ms,
            )

            if step % 10 == 0:
                logger.debug("Step %d: KV=%d bytes, latency=%.1fms", step, self._events[-1].kv_bytes_total, step_ms)

        else:
            # Reached max_new_tokens without EOS
            self._record_event(
                EventType.generation_end, step=max_new_tokens, prompt_tokens=prompt_len,
                generated_tokens=len(generated_ids), past_key_values=past_key_values,
            )

        logger.info(
            "Generation complete: %d tokens, peak KV cache: %d bytes",
            len(generated_ids), self._peak_kv,
        )

        return Trace(
            model_name=self.model_config.model_name,
            config=self.model_config,
            device=self.device,
            events=self._events,
        )

    def record_manual(
        self,
        event_type: EventType,
        step: int,
        prompt_tokens: int,
        generated_tokens: int,
        past_key_values: Any,
    ) -> None:
        """Manually record a KV cache event (for custom inference loops).

        Use this if you're not using generate() but want to instrument
        your own decode loop.
        """
        if not self._start_time_ms:
            self._start_time_ms = time.perf_counter() * 1000
        self._record_event(
            event_type, step, prompt_tokens, generated_tokens, past_key_values,
        )

    def get_trace(self) -> Trace:
        """Return the trace collected so far (for manual instrumentation)."""
        return Trace(
            model_name=self.model_config.model_name,
            config=self.model_config,
            device=self.device,
            events=self._events,
        )


def save_trace(trace: Trace, path: str | Path) -> Path:
    """Save a trace to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Trace saved to %s (%d events)", p, len(trace.events))
    return p


def load_trace(path: str | Path) -> Trace:
    """Load a trace from a JSON file."""
    p = Path(path)
    return Trace.model_validate_json(p.read_text(encoding="utf-8"))
