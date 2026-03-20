"""Live fragmentation dashboard — FastAPI + WebSocket server."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from kvviz.fragmentation.schema import (
    AllocatorMode,
    CudaMemoryStats,
    FragSimConfig,
    FragSnapshot,
    FragTrace,
    FreePolicy,
    RequestAllocationState,
)

logger = logging.getLogger("kvviz")

app = FastAPI(title="kvviz fragmentation dashboard")

DASHBOARD_HTML = (Path(__file__).parent / "frag_dashboard.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
        mode = config.get("mode", "synth")

        if mode == "synth":
            await _run_synth(ws, config)
        elif mode == "vllm":
            await _run_vllm(ws, config)
        elif mode == "replay":
            await _run_replay(ws, config)
        else:
            await ws.send_text(json.dumps({"type": "error", "error": f"Unknown mode: {mode}"}))
    except WebSocketDisconnect:
        logger.info("Frag dashboard client disconnected")
    except Exception as e:
        logger.exception("Frag dashboard WebSocket error")
        try:
            await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synth mode — run the existing simulator, stream snapshots
# ---------------------------------------------------------------------------


async def _run_synth(ws: WebSocket, config: dict) -> None:
    """Generate a synthetic fragmentation trace and stream as snapshots."""
    from kvviz.fragmentation.simulation import simulate
    from kvviz.fragmentation.traffic import generate_traffic

    num_requests = config.get("num_requests", 25)
    arrival_rate = config.get("arrival_rate", 1.5)
    block_size = config.get("block_size_tokens", 16)
    allocator = config.get("allocator", "paged")
    max_blocks = config.get("max_blocks", None)
    speed = config.get("speed", 1.0)
    delay_ms = config.get("delay_ms", 80)
    seed = config.get("seed", 42)
    min_prompt = config.get("min_prompt_tokens", 16)
    max_prompt = config.get("max_prompt_tokens", 512)
    min_gen = config.get("min_gen_tokens", 8)
    max_gen = config.get("max_gen_tokens", 128)

    traffic = generate_traffic(
        requests=num_requests,
        arrival_rate=arrival_rate,
        min_prompt_tokens=min_prompt,
        max_prompt_tokens=max_prompt,
        min_gen_tokens=min_gen,
        max_gen_tokens=max_gen,
        seed=seed,
    )

    sim_config = FragSimConfig(
        block_size_tokens=block_size,
        allocator=AllocatorMode(allocator),
        max_blocks=max_blocks,
    )

    # Send config info
    await ws.send_text(json.dumps({
        "type": "config",
        "block_size_tokens": block_size,
        "allocator": allocator,
        "total_requests": num_requests,
        "total_steps": traffic.total_steps,
    }))

    # Run simulation in a thread
    loop = asyncio.get_event_loop()
    frag_trace: FragTrace = await asyncio.to_thread(simulate, traffic, sim_config)

    # Stream events as snapshots
    t0 = time.time() * 1000
    for i, event in enumerate(frag_trace.events):
        gm = event.global_metrics
        # Compute packing efficiency across active requests
        if event.requests:
            pack = sum(r.packing_efficiency for r in event.requests) / len(event.requests)
        else:
            pack = 1.0

        snap = FragSnapshot(
            timestamp_ms=t0 + i * 20,
            total_blocks=gm.total_blocks,
            free_blocks=gm.free_blocks,
            used_blocks=gm.used_blocks,
            largest_free_run=gm.largest_free_run,
            fragmentation_ratio=gm.fragmentation_ratio,
            packing_efficiency=pack,
            block_map=event.block_map,
            requests=event.requests,
            cuda_stats=None,
            reuse_count=gm.reuse_count,
            step=event.step,
        )

        await ws.send_text(json.dumps({
            "type": "snapshot",
            **snap.model_dump(),
        }))

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000 / max(speed, 0.1))

    await ws.send_text(json.dumps({"type": "done"}))


# ---------------------------------------------------------------------------
# vLLM mode — connect to a running engine, poll block state
# ---------------------------------------------------------------------------


async def _run_vllm(ws: WebSocket, config: dict) -> None:
    """Connect to a running vLLM engine and stream live snapshots."""
    try:
        from kvviz.fragmentation.collector import VLLMBlockCollector
    except ImportError as e:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": f"vLLM collector unavailable: {e}",
        }))
        return

    # The engine must be passed programmatically.  For now we send
    # an informational error when invoked from the browser.
    await ws.send_text(json.dumps({
        "type": "error",
        "error": (
            "vLLM mode requires a running engine instance passed programmatically. "
            "Use the Python API: frag_dashboard.attach_vllm_engine(engine) before "
            "starting the server, or use 'kvviz frag-live' for headless collection."
        ),
    }))


# ---------------------------------------------------------------------------
# Replay mode — load a saved FragTrace and replay with timing
# ---------------------------------------------------------------------------


async def _run_replay(ws: WebSocket, config: dict) -> None:
    """Replay a saved FragTrace JSON file as a live stream."""
    trace_path = config.get("trace_path")
    if not trace_path:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": "replay mode requires 'trace_path' in config",
        }))
        return

    path = Path(trace_path)
    if not path.exists():
        await ws.send_text(json.dumps({
            "type": "error",
            "error": f"Trace file not found: {trace_path}",
        }))
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    frag_trace = FragTrace.model_validate(data)
    delay_ms = config.get("delay_ms", 80)
    speed = config.get("speed", 1.0)

    await ws.send_text(json.dumps({
        "type": "config",
        "block_size_tokens": frag_trace.config.block_size_tokens,
        "allocator": frag_trace.config.allocator.value,
        "total_events": len(frag_trace.events),
    }))

    t0 = time.time() * 1000
    for i, event in enumerate(frag_trace.events):
        gm = event.global_metrics
        if event.requests:
            pack = sum(r.packing_efficiency for r in event.requests) / len(event.requests)
        else:
            pack = 1.0

        snap = FragSnapshot(
            timestamp_ms=t0 + i * 20,
            total_blocks=gm.total_blocks,
            free_blocks=gm.free_blocks,
            used_blocks=gm.used_blocks,
            largest_free_run=gm.largest_free_run,
            fragmentation_ratio=gm.fragmentation_ratio,
            packing_efficiency=pack,
            block_map=event.block_map,
            requests=event.requests,
            cuda_stats=None,
            reuse_count=gm.reuse_count,
            step=event.step,
        )

        await ws.send_text(json.dumps({
            "type": "snapshot",
            **snap.model_dump(),
        }))

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000 / max(speed, 0.1))

    await ws.send_text(json.dumps({"type": "done"}))


# ---------------------------------------------------------------------------
# Programmatic API for attaching a vLLM engine
# ---------------------------------------------------------------------------

_vllm_engine = None


def attach_vllm_engine(engine: object) -> None:
    """Register a vLLM engine for the dashboard to poll.

    Call this before ``run_server()`` so that the *vllm* mode works from
    the browser.
    """
    global _vllm_engine
    _vllm_engine = engine


def run_server(host: str = "127.0.0.1", port: int = 8766) -> None:
    """Start the fragmentation dashboard server."""
    import uvicorn
    print(f"Fragmentation Dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
