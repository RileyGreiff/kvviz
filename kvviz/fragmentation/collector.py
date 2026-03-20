"""vLLM block state collector — polls a running engine for live KV cache state."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from kvviz.fragmentation.cuda_stats import capture_cuda_stats
from kvviz.fragmentation.schema import (
    CudaMemoryStats,
    FragSnapshot,
    RequestAllocationState,
)

logger = logging.getLogger("kvviz")


class VLLMBlockCollector:
    """Poll a vLLM engine's block manager for live fragmentation snapshots.

    The collector runs a background daemon thread at ``poll_hz`` and
    invokes ``on_snapshot`` with each :class:`FragSnapshot`.  It is
    version-tolerant: if a particular attribute path does not exist (e.g.
    across vLLM v0.6 / v0.7 / v0.8 internals), it degrades gracefully
    to reporting only free/total counts.

    Parameters
    ----------
    engine:
        A vLLM ``LLMEngine`` or ``AsyncLLMEngine`` instance.
    poll_hz:
        How many snapshots per second (default 5).
    block_size_tokens:
        Tokens per physical block (must match the engine config).
    on_snapshot:
        Callback invoked with each ``FragSnapshot``.
    cuda_device:
        Device index passed to :func:`capture_cuda_stats`.
    """

    def __init__(
        self,
        engine: Any,
        poll_hz: float = 5.0,
        block_size_tokens: int = 16,
        on_snapshot: Optional[Callable[[FragSnapshot], None]] = None,
        cuda_device: int = 0,
    ) -> None:
        self._engine = engine
        self._poll_hz = poll_hz
        self._block_size_tokens = block_size_tokens
        self._on_snapshot = on_snapshot
        self._cuda_device = cuda_device
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest: Optional[FragSnapshot] = None
        self._step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("VLLMBlockCollector started at %.1f Hz", self._poll_hz)

    def stop(self) -> None:
        """Stop the polling thread (blocks until join)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("VLLMBlockCollector stopped")

    def get_latest(self) -> Optional[FragSnapshot]:
        """Return the most recent snapshot (or ``None`` if not yet collected)."""
        return self._latest

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        interval = 1.0 / self._poll_hz
        while not self._stop_event.is_set():
            try:
                snap = self._collect()
                self._latest = snap
                if self._on_snapshot is not None:
                    self._on_snapshot(snap)
            except Exception:
                logger.exception("VLLMBlockCollector: error during collection tick")
            self._stop_event.wait(timeout=interval)

    def _collect(self) -> FragSnapshot:
        ts = time.time() * 1000
        self._step += 1

        # Try to reach the block manager through known attribute paths
        bm = self._resolve_block_manager()

        total_blocks = 0
        free_blocks = 0
        block_map: list[str | None] | None = None
        requests: list[RequestAllocationState] = []

        if bm is not None:
            total_blocks, free_blocks, block_map, requests = self._read_block_state(bm)

        used = total_blocks - free_blocks
        lfr = _largest_free_run(block_map) if block_map else 0
        frag = 1.0 - (lfr / free_blocks) if free_blocks > 0 else 0.0

        # Packing efficiency: average across active requests
        if requests:
            pack = sum(r.packing_efficiency for r in requests) / len(requests)
        else:
            pack = 1.0

        cuda = capture_cuda_stats(self._cuda_device)

        return FragSnapshot(
            timestamp_ms=ts,
            total_blocks=total_blocks,
            free_blocks=free_blocks,
            used_blocks=used,
            largest_free_run=lfr,
            fragmentation_ratio=frag,
            packing_efficiency=pack,
            block_map=block_map,
            requests=requests,
            cuda_stats=cuda,
            reuse_count=0,
            step=self._step,
        )

    # ------------------------------------------------------------------
    # vLLM version-tolerant attribute resolution
    # ------------------------------------------------------------------

    def _resolve_block_manager(self) -> Any:
        """Try multiple attribute paths to reach the block manager."""
        engine = self._engine
        # v0.6+: engine.scheduler[0].block_manager  (list of schedulers)
        # v0.5:  engine.scheduler.block_manager
        # AsyncLLMEngine wraps engine in .engine
        for root in (engine, getattr(engine, "engine", None)):
            if root is None:
                continue
            scheduler = getattr(root, "scheduler", None)
            if scheduler is None:
                continue
            # Could be a list of schedulers (pipeline parallel)
            if isinstance(scheduler, (list, tuple)):
                scheduler = scheduler[0] if scheduler else None
            if scheduler is None:
                continue
            bm = getattr(scheduler, "block_manager", None)
            if bm is not None:
                return bm
        return None

    def _read_block_state(self, bm: Any) -> tuple[
        int,
        int,
        list[str | None] | None,
        list[RequestAllocationState],
    ]:
        """Extract block state from a vLLM block manager.

        Returns (total_blocks, free_blocks, block_map, requests).
        """
        total = 0
        free = 0
        block_map: list[str | None] | None = None
        requests: list[RequestAllocationState] = []

        try:
            # v1 BlockSpaceManagerV1 / V2 expose gpu_allocator
            gpu_alloc = getattr(bm, "gpu_allocator", None)
            if gpu_alloc is not None:
                total = getattr(gpu_alloc, "num_blocks", 0)
                free_count = getattr(gpu_alloc, "get_num_free_blocks", None)
                if callable(free_count):
                    free = free_count()
                else:
                    free = getattr(gpu_alloc, "num_free_blocks", 0)
            else:
                # Fallback: some versions have get_num_free_gpu_blocks()
                gf = getattr(bm, "get_num_free_gpu_blocks", None)
                gt = getattr(bm, "get_num_total_gpu_blocks", None)
                if callable(gf):
                    free = gf()
                if callable(gt):
                    total = gt()
        except Exception:
            logger.debug("Could not read block counts from block manager", exc_info=True)

        # Attempt to read block tables for per-request detail
        try:
            block_tables = getattr(bm, "block_tables", None)
            if block_tables and isinstance(block_tables, dict):
                # block_tables: {seq_id: list[PhysicalBlock | int]}
                # Build a block_map
                if total > 0:
                    block_map = [None] * total
                for seq_id, blocks in block_tables.items():
                    req_id = str(seq_id)
                    num_blocks = len(blocks)
                    used_tokens = num_blocks * self._block_size_tokens
                    capacity = num_blocks * self._block_size_tokens
                    for blk in blocks:
                        idx = blk if isinstance(blk, int) else getattr(blk, "block_number", -1)
                        if block_map is not None and 0 <= idx < total:
                            block_map[idx] = req_id
                    eff = 1.0  # can't know exact tail waste without token count
                    requests.append(RequestAllocationState(
                        request_id=req_id,
                        live_tokens=used_tokens,
                        allocated_blocks=num_blocks,
                        allocated_capacity_tokens=capacity,
                        used_tokens=used_tokens,
                        tail_waste_tokens=0,
                        packing_efficiency=eff,
                    ))
        except Exception:
            logger.debug("Could not read block tables", exc_info=True)

        return total, free, block_map, requests


def _largest_free_run(block_map: list[str | None]) -> int:
    """Compute the longest contiguous run of free (``None``) blocks."""
    max_run = 0
    current = 0
    for b in block_map:
        if b is None:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run
