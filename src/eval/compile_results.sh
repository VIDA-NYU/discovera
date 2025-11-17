compile_results.py#!/bin/bash
# ==========================================================
# Purpose:
#   Wrapper to run the agreement summary analysis script
#   for multiple report types.
#
# Usage:
#   bash compile_results.sh
# ==========================================================

set -euo pipefail

# -----------------------------
# User-configurable parameters
# -----------------------------

BENCHMARK="../../data/benchmark/benchmark.csv"
EXPERIMENT="gpt-5_20251112_011802"
BASE_DIR="../../data"
REPORT_TYPES=(
  #"llm(o4-mini)" 
  #"discovera(gpt-4o)" 
  #"biomni(11-05-25)"
  "discovera(o4-mini)"
  #"biomni(o4-mini)"
  #"biomni(claude-3.5-haiku)"
  )

# -----------------------------
# Python script path
# -----------------------------
SCRIPT_PATH="./compile_results.py"

# -----------------------------
# Run the Python script
# -----------------------------
echo "===================================================="
echo " Running Agreement Summary Analysis "
echo "----------------------------------------------------"
echo " Benchmark path:    $BENCHMARK"
echo " Experiment ID:     $EXPERIMENT"
echo " Base directory:    $BASE_DIR"
echo " Report types:      ${REPORT_TYPES[*]}"
echo "===================================================="
echo

python "$SCRIPT_PATH" \
  --benchmark_path "$BENCHMARK" \
  --experiment_id "$EXPERIMENT" \
  --base_dir "$BASE_DIR" \
  --report_types "${REPORT_TYPES[@]}"

echo
echo "Analysis complete! Results saved under:"
echo "   $BASE_DIR/experiments/$EXPERIMENT/analysis/"
echo "===================================================="
