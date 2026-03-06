"""HTML report generator with rich KV cache visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kvviz.schema import EventType, Trace
from kvviz.utils import format_bytes

logger = logging.getLogger("kvviz")


def generate_report(trace: Trace, output_path: Path) -> Path:
    """Generate a standalone HTML report with interactive Plotly charts.

    Charts:
        1. KV Cache Memory Over Time (total + per-layer heatmap)
        2. Decode Latency Per Step (shows slowdown as context grows)
        3. Per-Layer KV Size Heatmap
        4. KV Bytes vs Sequence Length
        5. GPU Memory Timeline (if available)
        6. Summary stats table
    """
    if not trace.events:
        raise ValueError("Trace contains no events; cannot generate report.")

    decode_events = [e for e in trace.events if e.event_type == EventType.decode_step]
    all_timed = [e for e in trace.events if e.event_type in (EventType.prefill_end, EventType.decode_step)]

    has_gpu = any(e.gpu_allocated_bytes is not None for e in trace.events)
    has_layers = any(len(e.layers) > 0 for e in trace.events)

    # Determine subplot layout
    rows = 3  # always: memory timeline, latency, kv vs tokens
    if has_layers:
        rows += 1
    if has_gpu:
        rows += 1

    titles = ["KV Cache Memory Over Time", "Decode Latency Per Step", "KV Bytes vs Sequence Length"]
    if has_layers:
        titles.append("Per-Layer KV Cache Size")
    if has_gpu:
        titles.append("GPU Memory Timeline")

    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.06,
        specs=[[{"type": "scatter"}]] * (rows - (1 if has_layers else 0)) +
              ([[{"type": "heatmap"}]] if has_layers else []),
    )

    row = 1

    # --- Chart 1: KV Memory Over Time ---
    times = [e.timestamp_ms for e in all_timed]
    kv_bytes = [e.kv_bytes_total for e in all_timed]
    peak_bytes = [e.kv_bytes_peak for e in all_timed]

    fig.add_trace(go.Scatter(
        x=times, y=kv_bytes, mode="lines+markers",
        name="KV Current", line=dict(color="#636EFA"),
        hovertemplate="Time: %{x:.1f}ms<br>KV: %{customdata}<extra></extra>",
        customdata=[format_bytes(b) for b in kv_bytes],
    ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=times, y=peak_bytes, mode="lines",
        name="KV Peak", line=dict(color="#EF553B", dash="dash"),
    ), row=row, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=row, col=1)
    fig.update_yaxes(title_text="Bytes", row=row, col=1)

    row += 1

    # --- Chart 2: Decode Latency ---
    if decode_events:
        steps = [e.step for e in decode_events]
        latencies = [e.step_latency_ms for e in decode_events if e.step_latency_ms is not None]
        latency_steps = [e.step for e in decode_events if e.step_latency_ms is not None]

        fig.add_trace(go.Scatter(
            x=latency_steps, y=latencies, mode="lines+markers",
            name="Step Latency (ms)", line=dict(color="#00CC96"),
            hovertemplate="Step %{x}<br>Latency: %{y:.2f}ms<extra></extra>",
        ), row=row, col=1)
        fig.update_xaxes(title_text="Decode Step", row=row, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=row, col=1)

    row += 1

    # --- Chart 3: KV Bytes vs Sequence Length ---
    total_tokens = [e.total_tokens for e in all_timed]
    fig.add_trace(go.Scatter(
        x=total_tokens, y=kv_bytes, mode="markers",
        name="KV vs Tokens", marker=dict(color="#AB63FA"),
        hovertemplate="Tokens: %{x}<br>KV: %{customdata}<extra></extra>",
        customdata=[format_bytes(b) for b in kv_bytes],
    ), row=row, col=1)
    fig.update_xaxes(title_text="Total Tokens (prompt + generated)", row=row, col=1)
    fig.update_yaxes(title_text="KV Bytes", row=row, col=1)

    row += 1

    # --- Chart 4: GPU Memory (if available) ---
    if has_gpu:
        gpu_events = [e for e in trace.events if e.gpu_allocated_bytes is not None]
        gpu_times = [e.timestamp_ms for e in gpu_events]
        gpu_alloc = [e.gpu_allocated_bytes for e in gpu_events]
        gpu_reserved = [e.gpu_reserved_bytes or 0 for e in gpu_events]

        fig.add_trace(go.Scatter(
            x=gpu_times, y=gpu_alloc, mode="lines",
            name="GPU Allocated", line=dict(color="#FFA15A"),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=gpu_times, y=gpu_reserved, mode="lines",
            name="GPU Reserved", line=dict(color="#FF6692", dash="dot"),
        ), row=row, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=row, col=1)
        fig.update_yaxes(title_text="Bytes", row=row, col=1)
        row += 1

    # --- Chart 5: Per-Layer Heatmap ---
    if has_layers:
        events_with_layers = [e for e in all_timed if len(e.layers) > 0]
        if events_with_layers:
            num_layers = max(len(e.layers) for e in events_with_layers)
            z_data: list[list[float]] = []
            x_labels: list[str] = []

            for e in events_with_layers:
                x_labels.append(f"step {e.step}")
                row_data = [0.0] * num_layers
                for snap in e.layers:
                    if snap.layer_idx < num_layers:
                        row_data[snap.layer_idx] = snap.key_bytes + snap.value_bytes
                z_data.append(row_data)

            # Transpose: layers on Y axis, steps on X axis
            z_transposed = list(zip(*z_data)) if z_data else []

            fig.add_trace(go.Heatmap(
                z=z_transposed,
                x=x_labels,
                y=[f"Layer {i}" for i in range(num_layers)],
                colorscale="Viridis",
                name="Layer KV",
                colorbar=dict(title="Bytes"),
            ), row=row, col=1)

    # --- Layout ---
    height = 350 * rows
    fig.update_layout(
        title_text=f"KV Cache Monitor Report &mdash; {trace.model_name}",
        height=height,
        template="plotly_white",
        showlegend=True,
    )

    # --- Summary stats ---
    prefill_event = next((e for e in trace.events if e.event_type == EventType.prefill_end), None)
    end_event = next((e for e in trace.events if e.event_type == EventType.generation_end), None)

    prefill_ms = prefill_event.step_latency_ms if prefill_event else 0
    total_gen = end_event.generated_tokens if end_event else 0
    peak = trace.peak_kv_bytes

    avg_decode_ms = 0.0
    if decode_events:
        latencies_list = [e.step_latency_ms for e in decode_events if e.step_latency_ms is not None]
        avg_decode_ms = sum(latencies_list) / len(latencies_list) if latencies_list else 0

    tokens_per_sec = (1000 / avg_decode_ms) if avg_decode_ms > 0 else 0

    summary_html = f"""
    <h3>Summary</h3>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse; font-family:monospace; margin:1em 0;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Model</td><td>{trace.model_name}</td></tr>
      <tr><td>Device</td><td>{trace.device}</td></tr>
      <tr><td>Prompt Tokens</td><td>{prefill_event.prompt_tokens if prefill_event else 'N/A'}</td></tr>
      <tr><td>Generated Tokens</td><td>{total_gen}</td></tr>
      <tr><td>Prefill Latency</td><td>{prefill_ms:.1f} ms</td></tr>
      <tr><td>Avg Decode Latency</td><td>{avg_decode_ms:.2f} ms/token</td></tr>
      <tr><td>Throughput</td><td>{tokens_per_sec:.1f} tokens/sec</td></tr>
      <tr><td>Peak KV Cache</td><td>{format_bytes(peak)}</td></tr>
    </table>
    """

    if trace.config:
        cfg = trace.config
        kv_heads = cfg.num_kv_heads or cfg.num_attn_heads or "?"
        attn_heads = cfg.num_attn_heads or "?"
        gqa = ""
        if cfg.num_kv_heads and cfg.num_attn_heads and cfg.num_kv_heads != cfg.num_attn_heads:
            gqa = f" (GQA: {cfg.num_attn_heads}Q / {cfg.num_kv_heads}KV)"
        summary_html += f"""
    <h3>Model Config</h3>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse; font-family:monospace; margin:1em 0;">
      <tr><td>Layers</td><td>{cfg.num_layers}</td></tr>
      <tr><td>Attention Heads</td><td>{attn_heads}{gqa}</td></tr>
      <tr><td>KV Heads</td><td>{kv_heads}</td></tr>
      <tr><td>Head Dim</td><td>{cfg.head_dim}</td></tr>
      <tr><td>KV Dtype</td><td>{cfg.dtype.value}</td></tr>
    </table>
    """

    # Write standalone HTML
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>KV Cache Monitor Report - {trace.model_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2em; color: #333; }}
    h1 {{ border-bottom: 2px solid #636EFA; padding-bottom: 0.3em; }}
    h3 {{ color: #555; margin-top: 1.5em; }}
    table {{ border-color: #ddd; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>KV Cache Monitor Report</h1>
  <p>Generated from <strong>live monitoring</strong> of <strong>{trace.model_name}</strong> on <code>{trace.device}</code> | {len(trace.events)} events recorded</p>
  {chart_html}
  {summary_html}
  <p style="color:#999; margin-top:2em;">Generated by kvviz v0.2.0</p>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
