"""WebSocket integration tests for the fragmentation dashboard (synth mode)."""

from __future__ import annotations

import json

import pytest

from kvviz.frag_dashboard import app

# Requires httpx + anyio for async test client
pytest.importorskip("httpx")

from starlette.testclient import TestClient


def test_index_returns_html():
    """GET / serves the dashboard HTML page."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Live Fragmentation Dashboard" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_synth_mode_streams_snapshots():
    """Synth mode streams config, snapshots, and done messages."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({
            "mode": "synth",
            "num_requests": 5,
            "arrival_rate": 2.0,
            "block_size_tokens": 16,
            "delay_ms": 0,  # fast
            "speed": 10,
            "seed": 42,
        }))

        messages = []
        while True:
            data = json.loads(ws.receive_text())
            messages.append(data)
            if data.get("type") == "done":
                break
            # Safety valve
            if len(messages) > 2000:
                break

        types = [m["type"] for m in messages]
        assert types[0] == "config"
        assert "done" in types
        snapshot_msgs = [m for m in messages if m["type"] == "snapshot"]
        assert len(snapshot_msgs) > 0

        # Check snapshot fields
        s = snapshot_msgs[0]
        assert "total_blocks" in s
        assert "free_blocks" in s
        assert "used_blocks" in s
        assert "fragmentation_ratio" in s
        assert "block_map" in s
        assert "requests" in s


def test_unknown_mode_returns_error():
    """Unknown mode sends an error message."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"mode": "bogus"}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "error"
        assert "Unknown mode" in data["error"]


def test_replay_mode_missing_path():
    """Replay mode without trace_path sends an error."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"mode": "replay"}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "error"
        assert "trace_path" in data["error"]


def test_vllm_mode_no_engine():
    """vLLM mode without attached engine sends an informational error."""
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"mode": "vllm"}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "error"
        assert "engine" in data["error"].lower()
