"""Synthetic multi-request traffic generation."""

from __future__ import annotations

import random
from typing import Optional

from kvviz.fragmentation.schema import TrafficRequest, TrafficTrace


def generate_traffic(
    requests: int = 25,
    arrival_rate: float = 1.5,
    min_prompt_tokens: int = 16,
    max_prompt_tokens: int = 512,
    min_gen_tokens: int = 8,
    max_gen_tokens: int = 128,
    seed: Optional[int] = None,
) -> TrafficTrace:
    """Generate a synthetic multi-request traffic trace with overlapping requests.

    Requests arrive following a Poisson-like process with the given arrival rate
    (requests per time step). Each request has randomized prompt and generation
    lengths within the specified bounds.
    """
    if seed is not None:
        random.seed(seed)

    trace_requests: list[TrafficRequest] = []
    step = 0
    generated = 0

    while generated < requests:
        # Poisson-like: number of arrivals this step
        n_arrivals = 0
        while random.random() < (arrival_rate / (1 + arrival_rate)):
            n_arrivals += 1
            if generated + n_arrivals >= requests:
                break

        n_arrivals = max(1, n_arrivals) if generated == 0 and step == 0 else n_arrivals
        n_arrivals = min(n_arrivals, requests - generated)

        for _ in range(n_arrivals):
            prompt_len = random.randint(min_prompt_tokens, max_prompt_tokens)
            gen_len = random.randint(min_gen_tokens, max_gen_tokens)
            trace_requests.append(TrafficRequest(
                request_id=f"req_{generated:04d}",
                arrival_step=step,
                prompt_tokens=prompt_len,
                gen_tokens=gen_len,
            ))
            generated += 1

        step += 1

    # Compute total steps: last request arrival + its generation length
    total_steps = 0
    for r in trace_requests:
        end = r.arrival_step + r.gen_tokens
        total_steps = max(total_steps, end)

    return TrafficTrace(requests=trace_requests, total_steps=total_steps)
