# kvviz - KV Cache Visualizer

**Live monitoring and visualization of transformer KV cache memory during inference.**

kvviz hooks into your model's inference loop, measures the actual KV cache tensors at each decode step, and produces interactive HTML reports showing memory growth, per-layer breakdown, and decode latency. It also includes a real-time browser dashboard that streams KV cache data as tokens are generated.

## Why?

The KV cache is often the memory bottleneck during LLM inference. It grows linearly with sequence length and can silently consume most of your GPU. Existing tools show total GPU memory but don't tell you what's KV cache vs weights vs activations. kvviz fills that gap.

## Features

- **Live dashboard** — real-time browser UI with WebSocket streaming, updates charts as tokens are generated
- **Sliding-window eviction** — simulate KV cache eviction strategies and watch the cache plateau instead of growing forever
- **Live monitoring** — hooks into HuggingFace `generate()` and measures real KV cache tensor sizes at every decode step
- **Per-layer breakdown** — see which layers consume the most cache memory
- **Decode latency tracking** — visualize how generation slows down as context grows
- **GPU memory correlation** — track total GPU memory alongside KV cache
- **Offline estimation** — plan KV cache memory without a GPU
- **Standalone HTML reports** — interactive Plotly charts, viewable by double-clicking

## Quick Start

### Install

```bash
pip install -e .                # core (estimate + synth + report)
pip install -e ".[dashboard]"   # + live dashboard (FastAPI + WebSocket)
pip install -e ".[hf]"          # + torch + transformers for live model monitoring
pip install -e ".[dev]"         # + pytest for development
```

### Live Dashboard

```bash
kvviz dashboard
# Opens at http://127.0.0.1:8765
```

The dashboard supports two modes:

- **Synthetic** — no GPU needed, configurable model params, adjustable speed, instant demo
- **Live Model** — runs real HuggingFace inference with GPU data streaming in real-time

Set the **Cache Window** field to see sliding-window eviction in action — the KV cache will plateau at the window size instead of growing linearly.

### Monitor a Real Model

```bash
# Monitor GPT-2 (small, runs on CPU)
kvviz monitor \
  --model-name gpt2 \
  --prompt "The future of artificial intelligence is" \
  --max-new-tokens 100 \
  --out trace.json

# Generate interactive report
kvviz report trace.json --out report.html
# Open report.html in your browser
```

### Monitor with GPU

```bash
# Monitor Llama on GPU with fp16
kvviz monitor \
  --model-name meta-llama/Llama-2-7b-hf \
  --prompt "Explain KV caches in transformers:" \
  --max-new-tokens 200 \
  --dtype float16 \
  --out llama_trace.json

kvviz report llama_trace.json --out llama_report.html
```

### Estimate Without a GPU

```bash
kvviz estimate \
  --num-layers 32 --num-kv-heads 8 --num-attn-heads 32 \
  --head-dim 128 --dtype fp16 --seq-len 4096

# Estimate max context before OOM on 80GB GPU
kvviz estimate \
  --num-layers 80 --num-kv-heads 8 --head-dim 128 \
  --gpu-memory-gb 80 --reserved-gb 40
```

### Use in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvviz.tracker import KVCacheTracker, save_trace

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Basic monitoring
tracker = KVCacheTracker.from_model(model, tokenizer)
trace = tracker.generate("Hello world", max_new_tokens=50)
save_trace(trace, "trace.json")

# With sliding-window eviction (keep only last 64 tokens in cache)
tracker = KVCacheTracker.from_model(model, tokenizer, max_cache_tokens=64)
trace = tracker.generate("Hello world", max_new_tokens=100)
```

## What the Dashboard Shows

The live dashboard streams three real-time charts:

1. **KV Cache Memory** — current vs peak bytes, with eviction the current line plateaus while peak stays high
2. **Decode Latency** — per-step latency with running average, shows GPU jitter and context-length scaling
3. **Per-Layer KV Heatmap** — all layers updating each decode step

Plus live stat cards: decode step, KV size, peak, cache tokens, latency, throughput.

## What the Static Report Shows

1. **KV Cache Memory Over Time** — line chart of total KV bytes as generation progresses
2. **Decode Latency Per Step** — shows how each token gets slower as context grows
3. **KV Bytes vs Sequence Length** — confirms linear memory scaling
4. **Per-Layer KV Heatmap** — which layers use how much cache at each step
5. **GPU Memory Timeline** — total GPU allocation alongside KV cache (when on CUDA)
6. **Summary Table** — prompt tokens, generated tokens, peak KV, throughput

## KV Cache Math

```
KV bytes = num_layers * 2 * batch * seq_len * num_kv_heads * head_dim * bytes_per_elem
```

| Factor | Meaning |
|--------|---------|
| `num_layers` | Transformer layers (each stores K and V) |
| `2` | One tensor for Keys + one for Values |
| `batch` | Batch size |
| `seq_len` | Total tokens (prompt + generated so far) |
| `num_kv_heads` | Number of Key/Value heads |
| `head_dim` | Dimension per head |
| `bytes_per_elem` | fp16=2, bf16=2, fp32=4, fp8=1, int8=1 |

### GQA and MQA

**MHA** (Multi-Head Attention): `num_kv_heads == num_attn_heads`. Standard, most KV cache.

**GQA** (Grouped-Query Attention): `num_kv_heads < num_attn_heads`. Multiple Q heads share KV heads. Llama 2 70B: 64 Q heads, 8 KV heads = 8x savings.

**MQA** (Multi-Query Attention): `num_kv_heads == 1`. All Q heads share one KV head. Maximum savings.

kvviz measures the *actual* tensor sizes during inference, so GQA/MQA savings are reflected automatically in live monitoring.

## CLI Commands

| Command | Description |
|---------|-------------|
| `kvviz dashboard` | Launch the live real-time dashboard |
| `kvviz monitor` | Run monitored inference on a HuggingFace model |
| `kvviz estimate` | Estimate KV cache memory from architecture params |
| `kvviz synth-trace` | Generate synthetic trace for testing |
| `kvviz report` | Generate HTML report from a trace file |

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT
