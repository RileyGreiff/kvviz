"""CLI entry point using Typer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from kvviz.estimator import estimate_kv_cache, estimate_max_tokens
from kvviz.schema import DType, ModelConfig, RuntimeParams, Trace
from kvviz.synth import generate_trace
from kvviz.report import generate_report
from kvviz.tracker import save_trace, load_trace
from kvviz.utils import format_bytes, setup_logging

app = typer.Typer(
    name="kvviz",
    help="KV Cache Visualizer - live monitoring and visualization of transformer KV cache memory.",
    add_completion=False,
)
console = Console(stderr=True)
logger = logging.getLogger("kvviz")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging")):
    setup_logging(verbose)


# ---------- estimate ----------

@app.command()
def estimate(
    num_layers: int = typer.Option(..., help="Number of transformer layers"),
    head_dim: int = typer.Option(..., help="Dimension per attention head"),
    num_attn_heads: Optional[int] = typer.Option(None, help="Number of Q attention heads"),
    num_kv_heads: Optional[int] = typer.Option(None, help="Number of KV heads (GQA/MQA)"),
    dtype: DType = typer.Option(DType.fp16, help="Data type for KV cache"),
    batch: int = typer.Option(1, help="Batch size"),
    seq_len: int = typer.Option(512, help="Total sequence length"),
    config_json: Optional[Path] = typer.Option(None, help="Path to JSON model config"),
    model_name: Optional[str] = typer.Option(None, help="HuggingFace model name (requires transformers)"),
    emit_json: bool = typer.Option(False, "--json", help="Emit JSON to stdout"),
    gpu_memory_gb: Optional[float] = typer.Option(None, help="Total GPU memory in GB (for max-token estimate)"),
    reserved_gb: float = typer.Option(0.0, help="Reserved non-KV memory in GB"),
):
    """Estimate KV cache memory usage for a transformer model."""
    cfg_dict: dict = {}

    if config_json is not None:
        if not config_json.exists():
            console.print(f"[red]Config file not found: {config_json}[/red]")
            raise typer.Exit(1)
        cfg_dict = json.loads(config_json.read_text())

    if model_name is not None:
        try:
            from transformers import AutoConfig  # type: ignore
            hf_cfg = AutoConfig.from_pretrained(model_name)
            cfg_dict.setdefault("num_layers", getattr(hf_cfg, "num_hidden_layers", None))
            cfg_dict.setdefault("num_attn_heads", getattr(hf_cfg, "num_attention_heads", None))
            cfg_dict.setdefault("num_kv_heads", getattr(hf_cfg, "num_key_value_heads", None))
            head_dim_hf = getattr(hf_cfg, "head_dim", None)
            if head_dim_hf is None:
                hidden = getattr(hf_cfg, "hidden_size", None)
                nheads = getattr(hf_cfg, "num_attention_heads", None)
                if hidden and nheads:
                    head_dim_hf = hidden // nheads
            cfg_dict.setdefault("head_dim", head_dim_hf)
        except ImportError:
            console.print("[yellow]transformers not installed; ignoring --model-name[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load model config: {e}[/red]")
            raise typer.Exit(1)

    final_num_layers = cfg_dict.get("num_layers", num_layers)
    final_head_dim = cfg_dict.get("head_dim", head_dim)
    final_attn_heads = num_attn_heads or cfg_dict.get("num_attn_heads")
    final_kv_heads = num_kv_heads or cfg_dict.get("num_kv_heads")

    if final_kv_heads is None and final_attn_heads is None:
        console.print("[red]Must provide --num-kv-heads or --num-attn-heads[/red]")
        raise typer.Exit(1)

    config = ModelConfig(
        num_layers=final_num_layers,
        num_attn_heads=final_attn_heads,
        num_kv_heads=final_kv_heads,
        head_dim=final_head_dim,
        dtype=dtype,
    )
    params = RuntimeParams(batch=batch, seq_len=seq_len)
    result = estimate_kv_cache(config, params)

    if emit_json:
        d = result.as_dict()
        if gpu_memory_gb is not None:
            max_tok = estimate_max_tokens(
                config, int(gpu_memory_gb * 1024**3), int(reserved_gb * 1024**3), batch,
            )
            d["max_tokens_before_oom"] = max_tok
        sys.stdout.write(json.dumps(d, indent=2) + "\n")
        return

    tbl = Table(title="KV Cache Estimate", show_lines=True)
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value")
    tbl.add_row("Layers", str(result.num_layers))
    tbl.add_row("KV Heads", str(result.num_kv_heads))
    if final_attn_heads and final_attn_heads != result.num_kv_heads:
        tbl.add_row("Q Heads (GQA)", str(final_attn_heads))
    tbl.add_row("Head Dim", str(result.head_dim))
    tbl.add_row("Dtype", result.dtype)
    tbl.add_row("Batch", str(result.batch))
    tbl.add_row("Seq Length", str(result.seq_len))
    tbl.add_row("Total KV Cache", format_bytes(result.total_bytes))
    tbl.add_row("Per Token (1 batch elem)", format_bytes(result.bytes_per_token))
    tbl.add_row("Per Generated Token", format_bytes(result.bytes_per_generated_token))

    if gpu_memory_gb is not None:
        max_tok = estimate_max_tokens(
            config, int(gpu_memory_gb * 1024**3), int(reserved_gb * 1024**3), batch,
        )
        tbl.add_row("Max Tokens Before OOM", str(max_tok))

    console.print(tbl)


# ---------- monitor ----------

@app.command()
def monitor(
    model_name: str = typer.Option(..., help="HuggingFace model name or path"),
    prompt: str = typer.Option("The future of AI is", help="Input prompt text"),
    max_new_tokens: int = typer.Option(50, help="Maximum tokens to generate"),
    out: Path = typer.Option("trace.json", help="Output trace JSON path"),
    device: Optional[str] = typer.Option(None, help="Device (auto-detected if not set)"),
    dtype: Optional[str] = typer.Option(None, help="Model dtype: float16, bfloat16, float32, auto"),
):
    """Run monitored inference on a HuggingFace model and capture live KV cache data.

    Requires: pip install kvviz[hf]
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        console.print("[red]This command requires torch and transformers.[/red]")
        console.print("Install with: pip install kvviz[hf]")
        raise typer.Exit(1)

    from kvviz.tracker import KVCacheTracker

    console.print(f"Loading model [bold]{model_name}[/bold]...")

    load_kwargs: dict = {}
    if dtype:
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "auto": "auto",
        }
        if dtype in dtype_map:
            load_kwargs["torch_dtype"] = dtype_map[dtype]
        else:
            console.print(f"[yellow]Unknown dtype '{dtype}', using default[/yellow]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    console.print(f"Model loaded on [green]{device}[/green]")
    console.print(f"Generating up to {max_new_tokens} tokens...")

    tracker = KVCacheTracker.from_model(model, tokenizer, device=device)
    trace = tracker.generate(prompt, max_new_tokens=max_new_tokens)

    save_trace(trace, out)
    console.print(f"\nTrace saved to [green]{out}[/green] ({len(trace.events)} events)")
    console.print(f"Peak KV cache: [bold]{format_bytes(trace.peak_kv_bytes)}[/bold]")
    console.print(f"Generated {trace.total_generated} tokens")
    console.print(f"\nGenerate report with: [bold]kvviz report {out}[/bold]")


# ---------- synth-trace ----------

@app.command(name="synth-trace")
def synth_trace(
    out: Path = typer.Option("trace.json", help="Output trace JSON path"),
    num_layers: int = typer.Option(32, help="Number of layers"),
    num_kv_heads: int = typer.Option(8, help="Number of KV heads"),
    head_dim: int = typer.Option(128, help="Head dimension"),
    dtype: DType = typer.Option(DType.fp16, help="KV cache dtype"),
    prompt_tokens: int = typer.Option(128, help="Simulated prompt length"),
    max_new_tokens: int = typer.Option(64, help="Simulated generation length"),
    seed: Optional[int] = typer.Option(42, help="Random seed"),
):
    """Generate a synthetic KV cache trace (for testing without a GPU)."""
    trace = generate_trace(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    save_trace(trace, out)
    console.print(f"Synthetic trace written to [green]{out}[/green] ({len(trace.events)} events)")


# ---------- report ----------

@app.command()
def report(
    trace_path: Path = typer.Argument(..., help="Path to trace JSON"),
    out: Path = typer.Option("report.html", help="Output HTML report path"),
):
    """Generate a standalone HTML report from a trace file."""
    if not trace_path.exists():
        console.print(f"[red]Trace file not found: {trace_path}[/red]")
        raise typer.Exit(1)

    trace = load_trace(trace_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_path = generate_report(trace, out)
    console.print(f"Report written to [green]{result_path.resolve()}[/green]")


# ---------- dashboard ----------

@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8765, help="Port"),
):
    """Launch the live KV cache dashboard (WebSocket + browser UI)."""
    try:
        from kvviz.dashboard import run_server
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install fastapi uvicorn websockets")
        raise typer.Exit(1)
    run_server(host=host, port=port)


if __name__ == "__main__":
    app()
