#!/bin/bash
set -euo pipefail

# -----------------------------
# Parameters
# -----------------------------
EXPERIMENT="gpt-5_20250923_225017"
BENCHMARK="../../data/benchmark/benchmark.csv"
# Models/providers you want to test
MODELS=("gpt-5")

# Report sources to compare
BASES=("groundtruth")
COMPARES=(
    #"discovera(gpt-4o)" "llm(gpt-4o)"
    "biomni"
    )

# -----------------------------
# Run analysis for each combo
# -----------------------------
for BASE in "${BASES[@]}"; do
  for COMPARE in "${COMPARES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      echo ">>> Running analysis for base=$BASE vs compare=$COMPARE with model=$MODEL"

      python analyze_answers.py \
        --experiment "$EXPERIMENT" \
        --base "$BASE" \
        --compare "$COMPARE" \
        --model "$MODEL"
    done
  done
done

echo "All analyses completed!"

