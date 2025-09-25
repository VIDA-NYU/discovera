#!/bin/bash
set -euo pipefail

# ---------------------------
# Configuration
# ---------------------------
EXPERIMENT="gpt-5_20250923_225017"
BENCHMARK="../../data/benchmark/benchmark.csv"
PYTHON=python3  # or path to your venv python

# ---------------------------
# Reports
# ---------------------------
BASE="groundtruth"
COMPARES=(
    #"discovera(gpt-4o)" "llm(gpt-4o)"
    "biomni")

# ---------------------------
# Models
# ---------------------------
MODELS=("gpt-5")

# ---------------------------
# Run all comparisons
# ---------------------------
for MODEL in "${MODELS[@]}"; do
    for COMPARE in "${COMPARES[@]}"; do
        echo ">>> Running base=$BASE vs compare=$COMPARE with model=$MODEL"
        $PYTHON generate_answers.py \
            --experiment "$EXPERIMENT" \
            --benchmark "$BENCHMARK" \
            --base "$BASE" \
            --compare "$COMPARE" \
            --model "$MODEL"
    done
done

echo "All comparisons completed for all models."
