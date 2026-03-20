"""Fragmentation simulation driver."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from kvviz.fragmentation.allocator import BlockAllocator
from kvviz.fragmentation.metrics import compute_global_metrics, compute_request_metrics
from kvviz.fragmentation.schema import (
    AllocatorMode,
    FragEvent,
    FragEventType,
    FragSimConfig,
    FragTrace,
    FreePolicy,
    TrafficTrace,
)

logger = logging.getLogger("kvviz")


def _traffic_from_kvviz_trace(trace_data: dict) -> TrafficTrace:
    """Convert a kvviz monitor trace to a synthetic traffic trace.

    A kvviz trace represents a single request, so we create a traffic trace
    with one request whose token counts come from the trace events.
    """
    from kvviz.fragmentation.schema import TrafficRequest

    events = trace_data.get("events", [])
    prompt_tokens = 0
    gen_tokens = 0
    for e in events:
        if e.get("event_type") in ("prefill_end", "prefill_start"):
            prompt_tokens = max(prompt_tokens, e.get("prompt_tokens", 0))
        gen_tokens = max(gen_tokens, e.get("generated_tokens", 0))

    if prompt_tokens == 0:
        prompt_tokens = 128
    if gen_tokens == 0:
        gen_tokens = 64

    req = TrafficRequest(
        request_id="req_0000",
        arrival_step=0,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
    )
    return TrafficTrace(requests=[req], total_steps=gen_tokens)


def load_traffic(path: Path) -> tuple[TrafficTrace, str]:
    """Load a traffic trace or kvviz trace from a JSON file.

    Returns (traffic_trace, source_type) where source_type is
    'traffic' or 'kvviz_trace'.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Detect format: traffic traces have a "requests" key
    if "requests" in data and isinstance(data["requests"], list):
        if data["requests"] and "arrival_step" in data["requests"][0]:
            return TrafficTrace.model_validate(data), "traffic"

    # Assume kvviz trace format
    return _traffic_from_kvviz_trace(data), "kvviz_trace"


def simulate(
    traffic: TrafficTrace,
    config: FragSimConfig,
) -> FragTrace:
    """Run block allocation simulation over a traffic trace.

    Processes each request's lifecycle: arrival (prefill), per-token
    generation, and finish. At each event, records block allocation
    state and computes metrics.
    """
    allocator = BlockAllocator(
        block_size_tokens=config.block_size_tokens,
        max_blocks=config.max_blocks,
        mode=config.allocator,
    )

    events: list[FragEvent] = []
    event_idx = 0

    # Build timeline: for each step, which requests are active and what happens
    # active_requests: request_id -> {live_tokens, prompt_tokens, gen_tokens, gen_progress}
    active: dict[str, dict] = {}
    # Requests that finished but haven't been freed (for end_of_request policy)
    pending_free: list[str] = []

    for step in range(traffic.total_steps + 1):
        # Process arrivals at this step
        for req in traffic.requests:
            if req.arrival_step == step:
                # Request arrives: allocate blocks for prompt
                live_tokens = req.prompt_tokens
                if config.cache_window_tokens is not None:
                    live_tokens = min(live_tokens, config.cache_window_tokens)

                allocator.allocate(req.request_id, live_tokens)
                active[req.request_id] = {
                    "prompt_tokens": req.prompt_tokens,
                    "gen_tokens": req.gen_tokens,
                    "gen_progress": 0,
                    "live_tokens": live_tokens,
                    "total_tokens": req.prompt_tokens,
                }

                # Record arrival event
                req_states = _snapshot_requests(active, allocator, config)
                gm = compute_global_metrics(allocator)
                events.append(FragEvent(
                    event_idx=event_idx,
                    event_type=FragEventType.request_arrive,
                    request_id=req.request_id,
                    step=step,
                    requests=req_states,
                    global_metrics=gm,
                    block_map=allocator.snapshot_map(),
                ))
                event_idx += 1

        # Process token generation for active requests
        finished_this_step: list[str] = []
        for req_id, state in list(active.items()):
            if state["gen_progress"] < state["gen_tokens"]:
                state["gen_progress"] += 1
                state["total_tokens"] = state["prompt_tokens"] + state["gen_progress"]
                live = state["total_tokens"]
                if config.cache_window_tokens is not None:
                    live = min(live, config.cache_window_tokens)
                state["live_tokens"] = live

                allocator.allocate(req_id, live)

                if state["gen_progress"] >= state["gen_tokens"]:
                    finished_this_step.append(req_id)
                else:
                    # Record token generation event
                    req_states = _snapshot_requests(active, allocator, config)
                    gm = compute_global_metrics(allocator)
                    events.append(FragEvent(
                        event_idx=event_idx,
                        event_type=FragEventType.token_generate,
                        request_id=req_id,
                        step=step,
                        requests=req_states,
                        global_metrics=gm,
                        block_map=allocator.snapshot_map(),
                    ))
                    event_idx += 1

        # Process finished requests
        for req_id in finished_this_step:
            if config.free_policy == FreePolicy.immediate:
                allocator.free(req_id)
            else:
                pending_free.append(req_id)

            del active[req_id]

            req_states = _snapshot_requests(active, allocator, config)
            gm = compute_global_metrics(allocator)
            events.append(FragEvent(
                event_idx=event_idx,
                event_type=FragEventType.request_finish,
                request_id=req_id,
                step=step,
                requests=req_states,
                global_metrics=gm,
                block_map=allocator.snapshot_map(),
            ))
            event_idx += 1

    # Free any pending requests (end_of_request policy)
    for req_id in pending_free:
        allocator.free(req_id)

    logger.info(
        "Simulation complete: %d events, %d total blocks, peak used %d",
        len(events),
        allocator.total_blocks,
        max((e.global_metrics.used_blocks for e in events), default=0),
    )

    return FragTrace(config=config, events=events, source="simulation")


def _snapshot_requests(
    active: dict[str, dict],
    allocator: BlockAllocator,
    config: FragSimConfig,
) -> list:
    """Build per-request allocation snapshots for all active requests."""
    from kvviz.fragmentation.metrics import compute_request_metrics

    states = []
    for req_id, state in active.items():
        states.append(compute_request_metrics(
            req_id, state["live_tokens"], allocator,
        ))
    return states
