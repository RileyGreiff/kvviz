# PowerShell demo script for kvviz
# Run from the kvviz project root after: pip install -e .

$ErrorActionPreference = "Stop"

Write-Host "=== kvviz Demo ===" -ForegroundColor Cyan

Write-Host "`n--- Step 1: Estimate KV cache for a Llama-like model (GQA) ---" -ForegroundColor Yellow
kvviz estimate --num-layers 32 --num-kv-heads 8 --num-attn-heads 32 --head-dim 128 --dtype fp16 --batch 1 --seq-len 2048

Write-Host "`n--- Step 2: Generate synthetic trace (simulates live monitoring) ---" -ForegroundColor Yellow
$traceFile = Join-Path $PSScriptRoot ".." "demo_trace.json"
kvviz synth-trace --out $traceFile --num-layers 32 --num-kv-heads 8 --prompt-tokens 128 --max-new-tokens 64

Write-Host "`n--- Step 3: Generate HTML report ---" -ForegroundColor Yellow
$reportFile = Join-Path $PSScriptRoot ".." "demo_report.html"
kvviz report $traceFile --out $reportFile

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host "Trace:  $traceFile"
Write-Host "Report: $reportFile"
Write-Host "Open the report in your browser to view interactive charts."
Write-Host ""
Write-Host "To monitor a REAL model (requires torch + transformers):"
Write-Host "  kvviz monitor --model-name gpt2 --prompt 'Hello world' --max-new-tokens 50 --out real_trace.json"
Write-Host "  kvviz report real_trace.json --out real_report.html"
