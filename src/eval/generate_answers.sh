#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Config
# ---------------------------
EXPERIMENT_ID="${1:-}"  # Must be passed
COMPARES_STR="${2:-}"   # Optional: comma-separated list of compare reports
BENCHMARK="../../data/benchmark/benchmark.csv"
PYTHON=python3
BASE="groundtruth"
MODELS=("gpt-5")

# ---------------------------
# Check experiment ID
# ---------------------------
if [ -z "$EXPERIMENT_ID" ]; then
    echo "ERROR: You must provide an experiment ID as the first argument."
    echo "Usage: ./generate_answers.sh <EXPERIMENT_ID> [compare_reports]"
    exit 1
fi

# ---------------------------
# Determine compare reports
# ---------------------------
if [ -n "$COMPARES_STR" ]; then
    IFS=',' read -ra COMPARES <<< "$COMPARES_STR"
else
    # Infer compare reports from question filenames
    QUESTIONS_DIR="../../data/experiments/$EXPERIMENT_ID/questions"
    if [ ! -d "$QUESTIONS_DIR" ]; then
        echo "ERROR: Cannot find questions folder to infer compare reports."
        echo "       Please provide compare reports manually."
        exit 1
    fi

    COMPARES=()
    for f in "$QUESTIONS_DIR"/*.json; do
        fname=$(basename "$f" .json)
        # Extract substring after 'rs' and before '_openai'
        if [[ "$fname" =~ rs(.+)_openai ]]; then
            compare="${BASH_REMATCH[1]}"
            if [[ "$compare" != "groundtruth" ]]; then
                COMPARES+=("$compare")
            fi
        fi
    done

    if [ ${#COMPARES[@]} -eq 0 ]; then
        echo "ERROR: No compare reports found in $QUESTIONS_DIR."
        echo "       Please provide compare reports manually."
        exit 1
    fi
fi

# ---------------------------
# Run
# ---------------------------
for MODEL in "${MODELS[@]}"; do
    for COMPARE in "${COMPARES[@]}"; do
        # --- Print parameters before running ---
        echo "[INFO] Running generate_answers.py with parameters:"
        echo "       Benchmark: $BENCHMARK"
        echo "       Base report: $BASE"
        echo "       Compare report: $COMPARE"
        echo "       Model: $MODEL"
        echo "       Experiment ID: $EXPERIMENT_ID"
        echo ""

        # --- Run Python script ---
        $PYTHON generate_answers.py \
            --experiment "$EXPERIMENT_ID" \
            --benchmark "$BENCHMARK" \
            --base "$BASE" \
            --compare "$COMPARE" \
            --model "$MODEL"
    done
done


# #!/usr/bin/env bash
# set -euo pipefail

# # ---------------------------
# # Config (edit as needed)
# # ---------------------------
# EXPERIMENT_ID="${1:-}"  # Must be passed
# BENCHMARK="../../data/benchmark/benchmark.csv"
# PYTHON=python3
# BASE="groundtruth"
# COMPARES=("biomni(o4-mini)")
# MODELS=("gpt-5")

# # ---------------------------
# # Check experiment ID
# # ---------------------------
# if [ -z "$EXPERIMENT_ID" ]; then
#     echo "ERROR: You must provide an experiment ID as the first argument."
#     echo "Usage: ./generate_answers.sh <EXPERIMENT_ID>"
#     exit 1
# fi

# # ---------------------------
# # Run
# # ---------------------------
# for MODEL in "${MODELS[@]}"; do
#     for COMPARE in "${COMPARES[@]}"; do
#         echo ">>> Running generate_answers: base=$BASE vs compare=$COMPARE with model=$MODEL, experiment=$EXPERIMENT_ID"
#         $PYTHON generate_answers.py \
#             --experiment "$EXPERIMENT_ID" \
#             --benchmark "$BENCHMARK" \
#             --base "$BASE" \
#             --compare "$COMPARE" \
#             --model "$MODEL"
#     done
# done


# #!/bin/bash
# set -euo pipefail

# # ---------------------------
# # Configuration
# # ---------------------------
# EXPERIMENT="gpt-5_20251117_201135"
# BENCHMARK="../../data/benchmark/benchmark.csv"
# PYTHON=python3  # or path to your venv python

# # ---------------------------
# # Reports
# # ---------------------------
# BASE="groundtruth"
# COMPARES=(
#     #"discovera(gpt-4o)" 
#     #"llm(gpt-4o)"
#     #"llm(o4-mini)"
#     #"discovera(o4-mini)"
#     #"biomni"
#     #"biomni(11-05-25)"
#     "biomni(o4-mini)"
#     #"biomni(claude-3.5-haiku)"
#     )

# # ---------------------------
# # Models
# # ---------------------------
# MODELS=(
#     "gpt-5"
# )

# # ---------------------------
# # Run all comparisons
# # ---------------------------
# for MODEL in "${MODELS[@]}"; do
#     for COMPARE in "${COMPARES[@]}"; do
#         echo ">>> Running base=$BASE vs compare=$COMPARE with model=$MODEL"
#         $PYTHON generate_answers.py \
#             --experiment "$EXPERIMENT" \
#             --benchmark "$BENCHMARK" \
#             --base "$BASE" \
#             --compare "$COMPARE" \
#             --model "$MODEL"
#     done
# done

# echo "All comparisons completed for all models."
