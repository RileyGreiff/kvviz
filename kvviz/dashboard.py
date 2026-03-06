"""Live KV cache dashboard — FastAPI + WebSocket server."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from kvviz.schema import DType, Trace, TraceEvent
from kvviz.synth import generate_trace

logger = logging.getLogger("kvviz")

app = FastAPI(title="kvviz live dashboard")

DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # Wait for config from client
        raw = await ws.receive_text()
        config = json.loads(raw)
        mode = config.get("mode", "synth")

        if mode == "synth":
            await _run_synth(ws, config)
        elif mode == "live":
            await _run_live(ws, config)
        else:
            await ws.send_text(json.dumps({"error": f"Unknown mode: {mode}"}))
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


async def _run_synth(ws: WebSocket, config: dict) -> None:
    """Run synthetic trace generation, streaming events over WebSocket."""
    num_layers = config.get("num_layers", 32)
    num_kv_heads = config.get("num_kv_heads", 8)
    head_dim = config.get("head_dim", 128)
    dtype = DType(config.get("dtype", "fp16"))
    prompt_tokens = config.get("prompt_tokens", 128)
    max_new_tokens = config.get("max_new_tokens", 100)
    max_cache_tokens = config.get("max_cache_tokens", None)
    delay_ms = config.get("delay_ms", 50)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()

    def on_event(event: TraceEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    # Send model info
    await ws.send_text(json.dumps({
        "type": "config",
        "model_name": "synthetic",
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype.value,
        "max_cache_tokens": max_cache_tokens,
    }))

    # Generate in a thread so we don't block the event loop
    async def gen():
        await asyncio.to_thread(
            generate_trace,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_cache_tokens=max_cache_tokens,
            on_event=on_event,
        )
        await queue.put(None)  # sentinel

    gen_task = asyncio.create_task(gen())

    while True:
        event = await queue.get()
        if event is None:
            break
        await ws.send_text(json.dumps({
            "type": "event",
            **event.model_dump(),
        }))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

    await ws.send_text(json.dumps({"type": "done"}))
    await gen_task


async def _run_live(ws: WebSocket, config: dict) -> None:
    """Run live model inference, streaming events over WebSocket."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        await ws.send_text(json.dumps({
            "error": "Live mode requires torch and transformers. Install with: pip install kvviz[hf]"
        }))
        return

    from kvviz.tracker import KVCacheTracker

    model_name = config.get("model_name", "gpt2")
    prompt = config.get("prompt", "The future of AI is")
    max_new_tokens = config.get("max_new_tokens", 50)
    max_cache_tokens = config.get("max_cache_tokens", None)
    dtype_str = config.get("dtype", None)

    await ws.send_text(json.dumps({"type": "status", "message": f"Loading {model_name}..."}))

    load_kwargs: dict = {}
    if dtype_str:
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_str in dtype_map:
            load_kwargs["torch_dtype"] = dtype_map[dtype_str]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[TraceEvent | None] = asyncio.Queue()

    def on_event(event: TraceEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    tracker = KVCacheTracker.from_model(
        model, tokenizer, device=device,
        max_cache_tokens=max_cache_tokens, on_event=on_event,
    )

    # Send model config
    cfg = tracker.model_config
    await ws.send_text(json.dumps({
        "type": "config",
        "model_name": cfg.model_name,
        "num_layers": cfg.num_layers,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "dtype": cfg.dtype.value,
        "device": device,
        "max_cache_tokens": max_cache_tokens,
    }))

    await ws.send_text(json.dumps({"type": "status", "message": "Generating..."}))

    async def gen():
        await asyncio.to_thread(
            tracker.generate, prompt, max_new_tokens=max_new_tokens,
        )
        await queue.put(None)

    gen_task = asyncio.create_task(gen())

    while True:
        event = await queue.get()
        if event is None:
            break
        await ws.send_text(json.dumps({
            "type": "event",
            **event.model_dump(),
        }))

    await ws.send_text(json.dumps({"type": "done"}))
    await gen_task


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the dashboard server."""
    import uvicorn
    print(f"Dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
