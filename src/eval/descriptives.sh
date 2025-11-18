#!/usr/bin/env bash
# Simple runner for descriptives.py with output directory

# --- Configurable paths ---
BENCHMARK_PATH="../../data/benchmark/benchmark.csv"
OUTPUT_DIR="../../output/"
PROMPT_COL="Prompt"
CONTEXT_COL="Context/Background"

# --- Run Python analysis ---
python3 descriptives.py \
  --benchmark "$BENCHMARK_PATH" \
  --prompt "$PROMPT_COL" \
  --context "$CONTEXT_COL" \
  --output_dir "$OUTPUT_DIR"

echo " Benchmark analysis completed."
echo " Outputs saved in: $OUTPUT_DIR"
echo " - stats_bench_agg.csv"
echo " - stats_bench_dis.csv"
echo " - benchmark_analysis.log"
