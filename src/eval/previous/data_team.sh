#!/bin/bash

# Bash script to run the Python data processing script

# ---- Configuration ----
EXPERIMENT_ID="gpt-5_20250923_225017"
BASE_PATH="../../data/experiments"
OUTPUT_DIR="../../data/experiments"        # Optional: set to a custom output dir
QUIET=false          # Set to true to suppress output

# ---- Validation ----
if [ -z "$EXPERIMENT_ID" ]; then
  echo "Usage: $0 <experiment_id>"
  exit 1
fi

# ---- Command Construction ----
CMD="python3 data_team.py --experiment \"$EXPERIMENT_ID\" --base-path \"$BASE_PATH\""

if [ -n "$OUTPUT_DIR" ]; then
  CMD+=" --output-dir \"$OUTPUT_DIR\""
fi

if [ "$QUIET" = true ]; then
  CMD+=" --quiet"
fi

# ---- Run ----
echo "🚀 Running command:"
echo $CMD
eval $CMD
