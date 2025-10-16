#!/bin/bash
# ==========================================================
# Purpose:
#   Wrapper to run the semantic similarity analysis script.
#   Computes STS and ROUGE-L between model-generated outputs
#   and the specified ground truth column.
#
# Usage:
#   bash results.sh
# ==========================================================

# Exit immediately if a command exits with a non-zero status
set -e

# -----------------------------
# User-configurable parameters
# -----------------------------
INPUT_PATH="../../data/benchmark/benchmark.csv"
OUTPUT_DIR="../../output/"
STS_MODEL="dmlls/all-mpnet-base-v2-negation"
GROUND_TRUTH_COL="Ground Truth"
COMPARE_COLS=("LLM (gpt-4o)" "Discovera (gpt-4o)" "Biomni")

# -----------------------------
# Script location
# -----------------------------
SCRIPT_PATH="./results.py"

# -----------------------------
# Run the similarity script
# -----------------------------
echo "===================================================="
echo " Running Semantic Similarity Analysis "
echo "----------------------------------------------------"
echo " Input file:        $INPUT_PATH"
echo " Output directory:  $OUTPUT_DIR"
echo " Ground truth col:  $GROUND_TRUTH_COL"
echo " Compare columns:   ${COMPARE_COLS[*]}"
echo " Model:             $STS_MODEL"
echo "===================================================="
echo

python results.py \
  --input_path "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --sts_model "$STS_MODEL" \
  --ground_truth_col "$GROUND_TRUTH_COL" \
  --compare_cols "${COMPARE_COLS[@]}"

echo
echo "Analysis complete! Results saved to:"
echo "   $OUTPUT_DIR"
echo "===================================================="
