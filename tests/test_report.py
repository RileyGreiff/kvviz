"""Tests for report generation."""

from pathlib import Path

from kvviz.report import generate_report
from kvviz.synth import generate_trace


def test_report_creates_file(tmp_path: Path):
    trace = generate_trace(seed=42)
    out = tmp_path / "test_report.html"
    result = generate_report(trace, out)
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    assert "KV Cache" in content


def test_report_has_summary(tmp_path: Path):
    trace = generate_trace(seed=42)
    out = tmp_path / "report.html"
    generate_report(trace, out)
    content = out.read_text(encoding="utf-8")
    assert "Peak KV Cache" in content
    assert "Decode Latency" in content
    assert "synthetic" in content


def test_report_has_charts(tmp_path: Path):
    trace = generate_trace(num_layers=4, max_new_tokens=20, seed=42)
    out = tmp_path / "report.html"
    generate_report(trace, out)
    content = out.read_text(encoding="utf-8")
    assert "KV Cache Memory Over Time" in content
    assert "Decode Latency Per Step" in content
    assert "Per-Layer" in content
