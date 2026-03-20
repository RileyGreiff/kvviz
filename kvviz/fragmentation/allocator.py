"""Block allocators for KV cache fragmentation simulation."""

from __future__ import annotations

import math
from typing import Optional

from kvviz.fragmentation.schema import AllocatorMode


class BlockAllocator:
    """Base block allocator managing a fixed-size block pool.

    Each block holds ``block_size_tokens`` tokens. The block map tracks
    ownership: None = free, otherwise request_id string.
    """

    def __init__(
        self,
        block_size_tokens: int = 16,
        max_blocks: Optional[int] = None,
        mode: AllocatorMode = AllocatorMode.paged,
    ):
        self.block_size_tokens = block_size_tokens
        self.mode = mode
        self._max_blocks = max_blocks
        # block_map grows dynamically up to max_blocks
        self.block_map: list[Optional[str]] = []
        # Track which blocks belong to each request
        self._request_blocks: dict[str, list[int]] = {}
        # Metrics for current step
        self.blocks_allocated_this_step = 0
        self.blocks_freed_this_step = 0
        self.reuse_count = 0

    @property
    def total_blocks(self) -> int:
        return len(self.block_map)

    @property
    def free_blocks(self) -> int:
        return sum(1 for b in self.block_map if b is None)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    def _reset_step_metrics(self) -> None:
        self.blocks_allocated_this_step = 0
        self.blocks_freed_this_step = 0

    def _find_free_blocks(self, count: int) -> list[int]:
        """Find ``count`` free block indices. For paged mode, any free blocks.
        For contiguous mode, prefer a contiguous run."""
        free_indices = [i for i, b in enumerate(self.block_map) if b is None]

        if self.mode == AllocatorMode.contiguous and len(free_indices) >= count:
            # Try to find a contiguous run
            run_start = None
            run_len = 0
            for i in range(len(self.block_map)):
                if self.block_map[i] is None:
                    if run_start is None:
                        run_start = i
                        run_len = 1
                    else:
                        run_len += 1
                    if run_len >= count:
                        return list(range(run_start, run_start + count))
                else:
                    run_start = None
                    run_len = 0
            # Fall back to any free blocks if no contiguous run
            return free_indices[:count]

        return free_indices[:count]

    def _grow_pool(self, needed: int) -> None:
        """Expand block pool to accommodate ``needed`` additional blocks."""
        available_free = self.free_blocks
        to_add = needed - available_free
        if to_add <= 0:
            return
        if self._max_blocks is not None:
            capacity = self._max_blocks - self.total_blocks
            to_add = min(to_add, capacity)
        self.block_map.extend([None] * to_add)

    def allocate(self, request_id: str, tokens_needed: int) -> int:
        """Ensure ``request_id`` has enough blocks for ``tokens_needed``.

        Returns the number of blocks now allocated to this request.
        """
        self._reset_step_metrics()
        blocks_needed = math.ceil(tokens_needed / self.block_size_tokens)
        current_blocks = self._request_blocks.get(request_id, [])
        current_count = len(current_blocks)

        if blocks_needed <= current_count:
            return current_count

        additional = blocks_needed - current_count
        self._grow_pool(additional)

        new_blocks = self._find_free_blocks(additional)
        for idx in new_blocks:
            was_used = idx < len(self.block_map)  # reuse tracking
            if was_used and self.block_map[idx] is None:
                self.reuse_count += 1
            self.block_map[idx] = request_id
        self.blocks_allocated_this_step = len(new_blocks)

        if request_id not in self._request_blocks:
            self._request_blocks[request_id] = []
        self._request_blocks[request_id].extend(new_blocks)
        return len(self._request_blocks[request_id])

    def free(self, request_id: str) -> int:
        """Free all blocks belonging to ``request_id``. Returns count freed."""
        self._reset_step_metrics()
        blocks = self._request_blocks.pop(request_id, [])
        for idx in blocks:
            if idx < len(self.block_map):
                self.block_map[idx] = None
        self.blocks_freed_this_step = len(blocks)
        return len(blocks)

    def get_request_blocks(self, request_id: str) -> int:
        return len(self._request_blocks.get(request_id, []))

    def largest_free_run(self) -> int:
        """Length of the longest contiguous run of free blocks."""
        max_run = 0
        current_run = 0
        for b in self.block_map:
            if b is None:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def snapshot_map(self) -> list[Optional[str]]:
        """Return a copy of the current block map."""
        return list(self.block_map)
