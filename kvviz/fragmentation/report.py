"""HTML report generator for fragmentation simulation traces."""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kvviz.fragmentation.schema import FragEventType, FragTrace

logger = logging.getLogger("kvviz")

# Consistent color palette for requests
_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#D67195", "#79706E",
]


def _req_color(request_id: str, color_map: dict[str, str]) -> str:
    if request_id not in color_map:
        color_map[request_id] = _COLORS[len(color_map) % len(_COLORS)]
    return color_map[request_id]


def generate_frag_report(
    frag_trace: FragTrace,
    output_path: Path,
    compare_trace: FragTrace | None = None,
) -> Path:
    """Generate a standalone HTML fragmentation report.

    Charts:
        1. Block occupancy heatmap
        2. Fragmentation ratio over time
        3. Packing efficiency over time
        4. Free blocks and largest free run over time
        5. Per-request tail waste
        6. Block fill histogram
        7. Summary table
    """
    events = frag_trace.events
    if not events:
        raise ValueError("FragTrace contains no events; cannot generate report.")

    cfg = frag_trace.config
    color_map: dict[str, str] = {}

    # If comparing, build side-by-side; otherwise single
    traces_to_plot = [("", frag_trace)]
    if compare_trace is not None:
        traces_to_plot = [("Config A", frag_trace), ("Config B", compare_trace)]

    all_html_sections: list[str] = []

    for label, ft in traces_to_plot:
        evts = ft.events
        if not evts:
            continue

        section_id = label.replace(" ", "_").lower() or "main"
        prefix = f"{label}: " if label else ""

        # --- Chart 1: Block Occupancy Heatmap ---
        # Sample events to keep heatmap manageable
        sampled = _sample_events(evts, max_events=200)
        max_blocks = max(e.global_metrics.total_blocks for e in sampled) if sampled else 0

        if max_blocks > 0 and any(e.block_map for e in sampled):
            # Build heatmap data: rows = event index, cols = block id
            # value: request index (for coloring) or -1 for free
            all_req_ids: list[str] = []
            for e in sampled:
                if e.block_map:
                    for rid in e.block_map:
                        if rid and rid not in all_req_ids:
                            all_req_ids.append(rid)

            z_data: list[list[float]] = []
            hover_data: list[list[str]] = []
            y_labels: list[str] = []

            for e in sampled:
                y_labels.append(f"ev {e.event_idx}")
                row = []
                hover_row = []
                bmap = e.block_map or []
                for block_id in range(max_blocks):
                    if block_id < len(bmap) and bmap[block_id] is not None:
                        rid = bmap[block_id]
                        idx = all_req_ids.index(rid) + 1 if rid in all_req_ids else 0
                        row.append(idx)
                        # Find used tokens for this request
                        req_state = next(
                            (r for r in e.requests if r.request_id == rid), None
                        )
                        used_info = f"used: {req_state.used_tokens}t" if req_state else ""
                        hover_row.append(
                            f"block {block_id}<br>{rid}<br>{used_info}<br>event {e.event_idx}"
                        )
                    else:
                        row.append(0)
                        hover_row.append(f"block {block_id}<br>FREE<br>event {e.event_idx}")
                z_data.append(row)
                hover_data.append(hover_row)

            # Build discrete colorscale
            n_req = len(all_req_ids)
            colorscale = [[0, "#f0f0f0"]]  # free = light gray
            if n_req > 0:
                for i, rid in enumerate(all_req_ids):
                    val = (i + 1) / (n_req + 1)
                    colorscale.append([val, _req_color(rid, color_map)])
                colorscale.append([1.0, _COLORS[(n_req - 1) % len(_COLORS)]])

            heatmap_fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=[f"B{i}" for i in range(max_blocks)],
                y=y_labels,
                colorscale=colorscale,
                showscale=False,
                hovertext=hover_data,
                hoverinfo="text",
            ))
            heatmap_fig.update_layout(
                title=f"{prefix}Block Occupancy Over Time",
                xaxis_title="Block ID",
                yaxis_title="Event",
                height=max(400, min(len(sampled) * 4, 800)),
                template="plotly_white",
                yaxis=dict(autorange="reversed"),
            )
            all_html_sections.append(heatmap_fig.to_html(full_html=False, include_plotlyjs=False))

        # --- Charts 2-4: Time series ---
        event_indices = [e.event_idx for e in evts]
        frag_ratios = [e.global_metrics.fragmentation_ratio for e in evts]
        free_blocks = [e.global_metrics.free_blocks for e in evts]
        lfr = [e.global_metrics.largest_free_run for e in evts]
        used_blocks = [e.global_metrics.used_blocks for e in evts]

        # Packing efficiency: average across active requests
        packing_effs: list[float] = []
        for e in evts:
            if e.requests:
                avg_eff = sum(r.packing_efficiency for r in e.requests) / len(e.requests)
                packing_effs.append(avg_eff)
            else:
                packing_effs.append(1.0)

        ts_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f"{prefix}Fragmentation Ratio",
                f"{prefix}Packing Efficiency",
                f"{prefix}Free Blocks & Largest Free Run",
            ],
            vertical_spacing=0.08,
        )

        ts_fig.add_trace(go.Scatter(
            x=event_indices, y=frag_ratios, mode="lines",
            name="Fragmentation Ratio", line=dict(color="#EF553B"),
        ), row=1, col=1)
        ts_fig.update_yaxes(range=[0, 1.05], row=1, col=1)

        ts_fig.add_trace(go.Scatter(
            x=event_indices, y=packing_effs, mode="lines",
            name="Packing Efficiency", line=dict(color="#00CC96"),
        ), row=2, col=1)
        ts_fig.update_yaxes(range=[0, 1.05], row=2, col=1)

        ts_fig.add_trace(go.Scatter(
            x=event_indices, y=free_blocks, mode="lines",
            name="Free Blocks", line=dict(color="#636EFA"),
        ), row=3, col=1)
        ts_fig.add_trace(go.Scatter(
            x=event_indices, y=lfr, mode="lines",
            name="Largest Free Run", line=dict(color="#AB63FA", dash="dash"),
        ), row=3, col=1)

        ts_fig.update_layout(
            height=900, template="plotly_white", showlegend=True,
        )
        for r in range(1, 4):
            ts_fig.update_xaxes(title_text="Event Index", row=r, col=1)

        all_html_sections.append(ts_fig.to_html(full_html=False, include_plotlyjs=False))

        # --- Chart 5: Per-request tail waste ---
        # Collect max tail waste per request
        req_max_waste: dict[str, int] = defaultdict(int)
        for e in evts:
            for r in e.requests:
                req_max_waste[r.request_id] = max(
                    req_max_waste[r.request_id], r.tail_waste_tokens
                )

        if req_max_waste:
            waste_fig = go.Figure(data=go.Bar(
                x=list(req_max_waste.keys()),
                y=list(req_max_waste.values()),
                marker_color=[_req_color(r, color_map) for r in req_max_waste],
            ))
            waste_fig.update_layout(
                title=f"{prefix}Peak Tail Waste Per Request (tokens)",
                xaxis_title="Request",
                yaxis_title="Tail Waste (tokens)",
                height=350,
                template="plotly_white",
            )
            all_html_sections.append(waste_fig.to_html(full_html=False, include_plotlyjs=False))

        # --- Chart 6: Block fill histogram ---
        # From the last event with active requests, show how full each block is
        fill_values: list[float] = []
        for e in reversed(evts):
            if e.requests and e.block_map:
                bmap = e.block_map
                for r in e.requests:
                    if r.allocated_blocks > 0:
                        full_blocks = r.used_tokens // ft.config.block_size_tokens
                        partial = r.used_tokens % ft.config.block_size_tokens
                        for _ in range(full_blocks):
                            fill_values.append(1.0)
                        if partial > 0:
                            fill_values.append(partial / ft.config.block_size_tokens)
                break

        if fill_values:
            hist_fig = go.Figure(data=go.Histogram(
                x=fill_values,
                nbinsx=20,
                marker_color="#636EFA",
            ))
            hist_fig.update_layout(
                title=f"{prefix}Block Fill Rate Distribution",
                xaxis_title="Fill Rate",
                yaxis_title="Count",
                height=300,
                template="plotly_white",
            )
            all_html_sections.append(hist_fig.to_html(full_html=False, include_plotlyjs=False))

    # --- Summary table ---
    summary_html = _build_summary_table(frag_trace)
    if compare_trace is not None:
        summary_html += _build_summary_table(compare_trace, label="Config B")

    # --- Assemble HTML ---
    charts_html = "\n<hr>\n".join(all_html_sections)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>KV Cache Fragmentation Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2em; color: #333; }}
    h1 {{ border-bottom: 2px solid #636EFA; padding-bottom: 0.3em; }}
    h3 {{ color: #555; margin-top: 1.5em; }}
    table {{ border-collapse: collapse; font-family: monospace; margin: 1em 0; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f5f5f5; }}
    hr {{ border: none; border-top: 1px solid #eee; margin: 2em 0; }}
  </style>
</head>
<body>
  <h1>KV Cache Fragmentation Report</h1>
  <p>Block size: <strong>{frag_trace.config.block_size_tokens}</strong> tokens |
     Allocator: <strong>{frag_trace.config.allocator.value}</strong> |
     Free policy: <strong>{frag_trace.config.free_policy.value}</strong> |
     Events: <strong>{len(frag_trace.events)}</strong></p>
  {charts_html}
  {summary_html}
  <p style="color:#999; margin-top:2em;">Generated by kvviz v0.2.0</p>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Fragmentation report written to %s", output_path)
    return output_path


def _build_summary_table(ft: FragTrace, label: str = "") -> str:
    """Build an HTML summary table for a fragmentation trace."""
    evts = ft.events
    if not evts:
        return ""

    peak_used = max(e.global_metrics.used_blocks for e in evts)
    peak_total = max(e.global_metrics.total_blocks for e in evts)
    avg_frag = sum(e.global_metrics.fragmentation_ratio for e in evts) / len(evts)

    # Average packing efficiency across all events with active requests
    eff_vals = []
    for e in evts:
        for r in e.requests:
            eff_vals.append(r.packing_efficiency)
    avg_eff = sum(eff_vals) / len(eff_vals) if eff_vals else 1.0

    total_waste = sum(
        max((r.tail_waste_tokens for r in e.requests), default=0)
        for e in evts if e.requests
    )

    title = f"<h3>Summary{' — ' + label if label else ''}</h3>"
    return f"""{title}
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Block Size</td><td>{ft.config.block_size_tokens} tokens</td></tr>
      <tr><td>Allocator</td><td>{ft.config.allocator.value}</td></tr>
      <tr><td>Free Policy</td><td>{ft.config.free_policy.value}</td></tr>
      <tr><td>Peak Blocks Used</td><td>{peak_used}</td></tr>
      <tr><td>Total Blocks Allocated</td><td>{peak_total}</td></tr>
      <tr><td>Avg Fragmentation Ratio</td><td>{avg_frag:.3f}</td></tr>
      <tr><td>Avg Packing Efficiency</td><td>{avg_eff:.3f}</td></tr>
      <tr><td>Total Events</td><td>{len(evts)}</td></tr>
    </table>
    """


def _sample_events(events: list, max_events: int = 200) -> list:
    """Downsample events for heatmap readability."""
    if len(events) <= max_events:
        return events
    step = len(events) / max_events
    indices = [int(i * step) for i in range(max_events)]
    # Always include last event
    if indices[-1] != len(events) - 1:
        indices.append(len(events) - 1)
    return [events[i] for i in indices]
