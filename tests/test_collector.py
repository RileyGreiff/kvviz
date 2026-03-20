"""Tests for vLLM block state collector (mocked engine)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from kvviz.fragmentation.collector import VLLMBlockCollector, _largest_free_run
from kvviz.fragmentation.schema import FragSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_engine(
    total_blocks: int = 64,
    free_blocks: int = 20,
    block_tables: dict | None = None,
):
    """Create a mock vLLM engine with scheduler.block_manager."""
    gpu_alloc = MagicMock()
    gpu_alloc.num_blocks = total_blocks
    gpu_alloc.get_num_free_blocks.return_value = free_blocks

    bm = MagicMock()
    bm.gpu_allocator = gpu_alloc
    bm.block_tables = block_tables or {}

    scheduler = MagicMock()
    scheduler.block_manager = bm

    engine = MagicMock()
    engine.scheduler = scheduler
    # Remove .engine attr to avoid AsyncLLMEngine path confusion
    del engine.engine

    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collector_start_stop():
    """Collector starts a thread and stops cleanly."""
    engine = _make_mock_engine()
    collector = VLLMBlockCollector(engine, poll_hz=20)
    collector.start()
    time.sleep(0.15)  # Let it run a few ticks
    collector.stop()

    snap = collector.get_latest()
    assert snap is not None
    assert isinstance(snap, FragSnapshot)
    assert snap.total_blocks == 64
    assert snap.free_blocks == 20
    assert snap.used_blocks == 44


def test_collector_calls_callback():
    """on_snapshot callback receives snapshots."""
    engine = _make_mock_engine(total_blocks=32, free_blocks=10)
    received = []

    collector = VLLMBlockCollector(
        engine, poll_hz=50, on_snapshot=received.append,
    )
    collector.start()
    time.sleep(0.12)
    collector.stop()

    assert len(received) >= 2
    for snap in received:
        assert isinstance(snap, FragSnapshot)
        assert snap.total_blocks == 32


def test_collector_with_block_tables():
    """Collector extracts per-request state from block_tables."""
    block_tables = {
        "seq_001": [0, 3, 7],
        "seq_002": [1, 4],
    }
    engine = _make_mock_engine(
        total_blocks=16, free_blocks=11, block_tables=block_tables,
    )

    collector = VLLMBlockCollector(engine, poll_hz=100, block_size_tokens=16)
    collector.start()
    time.sleep(0.08)
    collector.stop()

    snap = collector.get_latest()
    assert snap is not None
    assert len(snap.requests) == 2
    req_ids = {r.request_id for r in snap.requests}
    assert "seq_001" in req_ids
    assert "seq_002" in req_ids


def test_collector_no_block_manager():
    """Gracefully handles engine without a block manager."""
    engine = MagicMock()
    engine.scheduler = None
    del engine.engine

    collector = VLLMBlockCollector(engine, poll_hz=50)
    collector.start()
    time.sleep(0.08)
    collector.stop()

    snap = collector.get_latest()
    assert snap is not None
    assert snap.total_blocks == 0
    assert snap.free_blocks == 0


def test_collector_async_engine_wrapper():
    """Handles AsyncLLMEngine that wraps engine in .engine attribute."""
    inner_engine = _make_mock_engine(total_blocks=100, free_blocks=50)
    outer = MagicMock()
    outer.scheduler = None
    outer.engine = inner_engine

    collector = VLLMBlockCollector(outer, poll_hz=50)
    collector.start()
    time.sleep(0.08)
    collector.stop()

    snap = collector.get_latest()
    assert snap is not None
    assert snap.total_blocks == 100
    assert snap.free_blocks == 50


def test_collector_scheduler_list():
    """Handles engine.scheduler as a list (pipeline parallel)."""
    bm = MagicMock()
    gpu_alloc = MagicMock()
    gpu_alloc.num_blocks = 48
    gpu_alloc.get_num_free_blocks.return_value = 12
    bm.gpu_allocator = gpu_alloc
    bm.block_tables = {}

    scheduler = MagicMock()
    scheduler.block_manager = bm

    engine = MagicMock()
    engine.scheduler = [scheduler]
    del engine.engine

    collector = VLLMBlockCollector(engine, poll_hz=50)
    collector.start()
    time.sleep(0.08)
    collector.stop()

    snap = collector.get_latest()
    assert snap is not None
    assert snap.total_blocks == 48


# ---------------------------------------------------------------------------
# _largest_free_run
# ---------------------------------------------------------------------------


def test_largest_free_run_empty():
    assert _largest_free_run([]) == 0


def test_largest_free_run_all_free():
    assert _largest_free_run([None, None, None]) == 3


def test_largest_free_run_no_free():
    assert _largest_free_run(["a", "b", "c"]) == 0


def test_largest_free_run_mixed():
    bmap = ["a", None, None, "b", None, None, None, "c"]
    assert _largest_free_run(bmap) == 3
