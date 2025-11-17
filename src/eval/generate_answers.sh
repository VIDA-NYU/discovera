#!/bin/bash
set -euo pipefail

# ---------------------------
# Configuration
# ---------------------------
EXPERIMENT="gpt-5_20251114_005339"
BENCHMARK="../../data/benchmark/benchmark.csv"
PYTHON=python3  # or path to your venv python

# ---------------------------
# Reports
# ---------------------------
BASE="groundtruth"
COMPARES=(
    #"discovera(gpt-4o)" 
    #"llm(gpt-4o)"
    #"llm(o4-mini)"
    #"discovera(o4-mini)"
    #"biomni"
    #"biomni(11-05-25)"
    "biomni(o4-mini)"
    "biomni(claude-3.5-haiku)"
    )

# ---------------------------
# Models
# ---------------------------
MODELS=(
    "gpt-5"
    #"gpt-o4"
)

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
