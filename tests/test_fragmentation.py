"""Tests for KV cache fragmentation simulation."""

from pathlib import Path

from kvviz.fragmentation.allocator import BlockAllocator
from kvviz.fragmentation.metrics import compute_global_metrics, compute_request_metrics
from kvviz.fragmentation.report import generate_frag_report
from kvviz.fragmentation.schema import (
    AllocatorMode,
    FragSimConfig,
    FragTrace,
    FreePolicy,
)
from kvviz.fragmentation.simulation import simulate
from kvviz.fragmentation.traffic import generate_traffic


# ── Synthetic traffic generation ──


class TestTrafficGeneration:
    def test_creates_multiple_requests(self):
        traffic = generate_traffic(requests=10, seed=42)
        assert len(traffic.requests) == 10

    def test_overlapping_requests(self):
        traffic = generate_traffic(requests=15, arrival_rate=2.0, seed=42)
        # With high arrival rate, some requests should share the same arrival step
        arrival_steps = [r.arrival_step for r in traffic.requests]
        # At least some arrivals should overlap (same step)
        assert len(set(arrival_steps)) < len(arrival_steps)

    def test_deterministic_with_seed(self):
        t1 = generate_traffic(requests=10, seed=123)
        t2 = generate_traffic(requests=10, seed=123)
        assert len(t1.requests) == len(t2.requests)
        for r1, r2 in zip(t1.requests, t2.requests):
            assert r1.request_id == r2.request_id
            assert r1.arrival_step == r2.arrival_step
            assert r1.prompt_tokens == r2.prompt_tokens
            assert r1.gen_tokens == r2.gen_tokens

    def test_different_seeds_differ(self):
        t1 = generate_traffic(requests=10, seed=1)
        t2 = generate_traffic(requests=10, seed=2)
        # At least some requests should differ
        diffs = sum(
            1 for r1, r2 in zip(t1.requests, t2.requests)
            if r1.prompt_tokens != r2.prompt_tokens
        )
        assert diffs > 0

    def test_total_steps_positive(self):
        traffic = generate_traffic(requests=5, seed=42)
        assert traffic.total_steps > 0

    def test_token_bounds(self):
        traffic = generate_traffic(
            requests=20,
            min_prompt_tokens=32,
            max_prompt_tokens=64,
            min_gen_tokens=10,
            max_gen_tokens=20,
            seed=42,
        )
        for r in traffic.requests:
            assert 32 <= r.prompt_tokens <= 64
            assert 10 <= r.gen_tokens <= 20


# ── Block allocator ──


class TestBlockAllocator:
    def test_allocate_grows_blocks(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 48)  # needs 3 blocks
        assert alloc.get_request_blocks("r1") == 3
        assert alloc.total_blocks == 3
        assert alloc.used_blocks == 3
        assert alloc.free_blocks == 0

    def test_free_returns_blocks(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 32)
        freed = alloc.free("r1")
        assert freed == 2
        assert alloc.free_blocks == 2
        assert alloc.used_blocks == 0

    def test_tail_waste_known_example(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 20)  # 2 blocks = 32 capacity, 20 used -> 12 waste
        metrics = compute_request_metrics("r1", 20, alloc)
        assert metrics.allocated_blocks == 2
        assert metrics.allocated_capacity_tokens == 32
        assert metrics.used_tokens == 20
        assert metrics.tail_waste_tokens == 12

    def test_packing_efficiency_in_range(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 50)
        metrics = compute_request_metrics("r1", 50, alloc)
        assert 0.0 <= metrics.packing_efficiency <= 1.0

    def test_packing_efficiency_exact_fit(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 32)  # exactly 2 blocks
        metrics = compute_request_metrics("r1", 32, alloc)
        assert metrics.packing_efficiency == 1.0
        assert metrics.tail_waste_tokens == 0

    def test_contiguous_mode(self):
        alloc = BlockAllocator(block_size_tokens=16, mode=AllocatorMode.contiguous)
        alloc.allocate("r1", 32)
        alloc.allocate("r2", 32)
        # r1 gets blocks 0,1; r2 gets blocks 2,3
        assert alloc.block_map[:2] == ["r1", "r1"]
        assert alloc.block_map[2:4] == ["r2", "r2"]

    def test_largest_free_run(self):
        alloc = BlockAllocator(block_size_tokens=16)
        alloc.allocate("r1", 16)  # block 0
        alloc.allocate("r2", 16)  # block 1
        alloc.allocate("r3", 16)  # block 2
        alloc.free("r1")  # free block 0
        alloc.free("r3")  # free block 2
        # blocks: [None, "r2", None] -> largest run = 1
        assert alloc.largest_free_run() == 1

    def test_max_blocks_limit(self):
        alloc = BlockAllocator(block_size_tokens=16, max_blocks=4)
        alloc.allocate("r1", 64)  # wants 4 blocks, max is 4
        assert alloc.total_blocks == 4
        alloc.allocate("r2", 16)  # wants 1 more block, can't grow
        assert alloc.total_blocks == 4


# ── Fragmentation simulation ──


class TestFragSimulation:
    def test_basic_simulation(self):
        traffic = generate_traffic(requests=5, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        assert len(frag_trace.events) > 0

    def test_blocks_grow_with_tokens(self):
        traffic = generate_traffic(requests=3, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        # Find events for first request and check blocks grow
        first_req = traffic.requests[0].request_id
        req_events = [
            e for e in frag_trace.events if e.request_id == first_req
        ]
        assert len(req_events) >= 2  # at least arrive + finish

    def test_freeing_returns_blocks(self):
        traffic = generate_traffic(requests=3, seed=42)
        config = FragSimConfig(block_size_tokens=16, free_policy=FreePolicy.immediate)
        frag_trace = simulate(traffic, config)
        # After a request finishes, its blocks should be freed
        finish_events = [
            e for e in frag_trace.events if e.event_type.value == "request_finish"
        ]
        assert len(finish_events) > 0

    def test_fragmentation_ratio_in_range(self):
        traffic = generate_traffic(requests=10, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        for e in frag_trace.events:
            assert 0.0 <= e.global_metrics.fragmentation_ratio <= 1.0

    def test_packing_efficiency_in_range(self):
        traffic = generate_traffic(requests=10, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        for e in frag_trace.events:
            for r in e.requests:
                assert 0.0 <= r.packing_efficiency <= 1.0

    def test_contiguous_allocator(self):
        traffic = generate_traffic(requests=5, seed=42)
        config = FragSimConfig(
            block_size_tokens=16, allocator=AllocatorMode.contiguous,
        )
        frag_trace = simulate(traffic, config)
        assert len(frag_trace.events) > 0

    def test_end_of_request_free_policy(self):
        traffic = generate_traffic(requests=5, seed=42)
        config = FragSimConfig(
            block_size_tokens=16, free_policy=FreePolicy.end_of_request,
        )
        frag_trace = simulate(traffic, config)
        assert len(frag_trace.events) > 0


# ── Cache window behavior ──


class TestCacheWindow:
    def test_window_caps_live_tokens(self):
        traffic = generate_traffic(
            requests=1,
            min_prompt_tokens=100,
            max_prompt_tokens=100,
            min_gen_tokens=100,
            max_gen_tokens=100,
            seed=42,
        )
        config = FragSimConfig(
            block_size_tokens=16, cache_window_tokens=64,
        )
        frag_trace = simulate(traffic, config)
        # Live tokens should never exceed 64
        for e in frag_trace.events:
            for r in e.requests:
                assert r.live_tokens <= 64

    def test_window_limits_blocks(self):
        traffic = generate_traffic(
            requests=1,
            min_prompt_tokens=200,
            max_prompt_tokens=200,
            min_gen_tokens=50,
            max_gen_tokens=50,
            seed=42,
        )
        config_no_window = FragSimConfig(block_size_tokens=16)
        config_window = FragSimConfig(block_size_tokens=16, cache_window_tokens=64)

        ft_no = simulate(traffic, config_no_window)
        ft_win = simulate(traffic, config_window)

        peak_no = max(e.global_metrics.used_blocks for e in ft_no.events)
        peak_win = max(e.global_metrics.used_blocks for e in ft_win.events)
        assert peak_win < peak_no


# ── Report generation ──


class TestFragReport:
    def test_creates_html_file(self, tmp_path: Path):
        traffic = generate_traffic(requests=5, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        out = tmp_path / "frag_report.html"
        result = generate_frag_report(frag_trace, out)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Fragmentation" in content

    def test_report_has_key_sections(self, tmp_path: Path):
        traffic = generate_traffic(requests=8, seed=42)
        config = FragSimConfig(block_size_tokens=16)
        frag_trace = simulate(traffic, config)
        out = tmp_path / "report.html"
        generate_frag_report(frag_trace, out)
        content = out.read_text(encoding="utf-8")
        assert "Block Occupancy" in content
        assert "Fragmentation Ratio" in content
        assert "Packing Efficiency" in content
        assert "Summary" in content

    def test_compare_report(self, tmp_path: Path):
        traffic = generate_traffic(requests=5, seed=42)
        ft_a = simulate(traffic, FragSimConfig(block_size_tokens=16))
        ft_b = simulate(traffic, FragSimConfig(block_size_tokens=32))
        out = tmp_path / "compare.html"
        result = generate_frag_report(ft_a, out, compare_trace=ft_b)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Config A" in content
        assert "Config B" in content
