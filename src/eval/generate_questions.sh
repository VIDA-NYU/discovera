#!/usr/bin/env bash
# Bash wrapper to run generate_questions.py with multiple settings

set -euo pipefail

# ------------------------
# Config (edit as needed)
# ------------------------
BENCHMARK="../../data/benchmark/benchmark.csv"
OUTDIR="../../data/experiments/"
NS=10 # Number of samples per question

# List of models to try
MODELS=("gpt-5")

# List of report columns
REPORT_COLUMNS=(
  #"LLM (gpt-4o)" 
  #"LLM (o4-mini)"
  #"Discovera (gpt-4o)" 
  #"Discovera (o4-mini)"
  #"Ground Truth"
  #"Biomni"
  #"Biomni (11-05-25)"
  "Biomni (o4-mini)"
  )

# ------------------------
# Run
# ------------------------
for MODEL in "${MODELS[@]}"; do
  echo "=== Running model: $MODEL ==="
  
  # Join report columns with quotes for Python
  COLS=()
  for COL in "${REPORT_COLUMNS[@]}"; do
    COLS+=("$COL")
  done
  
  python3 generate_questions.py \
    --benchmark "$BENCHMARK" \
    --report-columns "${COLS[@]}" \
    --ns "$NS" \
    --model "$MODEL" \
    --out "$OUTDIR"
  
done
