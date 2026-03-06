#!/usr/bin/env bash
# Demo script for kvviz
# Run from the kvviz project root after: pip install -e .
set -euo pipefail

echo "=== kvviz Demo ==="

echo ""
echo "--- Step 1: Estimate KV cache for a Llama-like model (GQA) ---"
kvviz estimate \
  --num-layers 32 \
  --num-kv-heads 8 \
  --num-attn-heads 32 \
  --head-dim 128 \
  --dtype fp16 \
  --batch 1 \
  --seq-len 2048

echo ""
echo "--- Step 2: Generate synthetic trace (simulates live monitoring) ---"
kvviz synth-trace --out demo_trace.json \
  --num-layers 32 --num-kv-heads 8 \
  --prompt-tokens 128 --max-new-tokens 64

echo ""
echo "--- Step 3: Generate HTML report ---"
kvviz report demo_trace.json --out demo_report.html

echo ""
echo "=== Done! ==="
echo "Trace:  demo_trace.json"
echo "Report: demo_report.html"
echo "Open the report in your browser to view interactive charts."
echo ""
echo "To monitor a REAL model (requires torch + transformers):"
echo "  kvviz monitor --model-name gpt2 --prompt 'Hello world' --max-new-tokens 50 --out real_trace.json"
echo "  kvviz report real_trace.json --out real_report.html"
