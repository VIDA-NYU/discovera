#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Config (edit as needed)
# ---------------------------
BENCHMARK="../../data/benchmark/benchmark.csv"
OUTDIR="../../data/experiments/"
NS=2  # Number of samples per question
MODELS=("gpt-5")  # Models to run
REPORT_COLUMNS=("Ground Truth" "Biomni (o4-mini)")  # Reports to generate questions for
TASK_IDS="1,2.1,10"  # Optional: tasks to run
PYTHON=python3  # Python interpreter
BASE="groundtruth"

# ---------------------------
# Retrieve traditional similarity metrics
# ---------------------------
echo "=================================================="
echo "[MASTER] Step 0: 
    - Run traditional similarity analysis...
    - Get descriptives of benchmark data..."
echo "=================================================="
bash similarity_metrics.sh
echo "[MASTER] Semantic similarity analysis completed!"
echo ""

bash descriptives.sh
echo "[MASTER] Descriptives retrieved!"
echo ""

# ---------------------------
# Run pipeline for report similarity based on LLMs for each model
# ---------------------------
for MODEL in "${MODELS[@]}"; do
    # Generate timestamped experiment ID
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_ID="${MODEL}_${TIMESTAMP}"
    
    echo "=================================================="
    echo "[MASTER] Running full pipeline for model: $MODEL"
    echo "[MASTER] Experiment ID: $EXPERIMENT_ID"
    echo "=================================================="
    echo ""

    # ---------------------------
    # Step 1: Generate questions
    # ---------------------------
    echo "[MASTER] Step 1: Generating questions..."
    ./generate_questions.sh "$EXPERIMENT_ID"
    echo "[MASTER] Questions generated for experiment: $EXPERIMENT_ID"
    echo ""

    # ---------------------------
    # Step 2: Generate answers
    # ---------------------------
    echo "[MASTER] Step 2: Generating answers..."
    ./generate_answers.sh "$EXPERIMENT_ID"
    echo "[MASTER] Answers generated for experiment: $EXPERIMENT_ID"
    echo ""
done

echo "[MASTER] All experiments completed!"
