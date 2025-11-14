#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------

PYTHON_SCRIPT="biomni_reports.py"

BENCHMARK="../../data/benchmark/benchmark.csv"
TASKS_FOLDER="../../data/benchmark/tasks"
LOGS_FOLDER="../../biomni_logs"
OUTPUT_CSV="../../biomni_results.csv"

MODEL="OpenAI O4-Mini"

# Optional experiment name
EXPERIMENT="${1:-biomni_run_$(date +%Y%m%d_%H%M%S)}"

echo "==========================================="
echo "   Retrieving Biomni Reports Processing"
echo "==========================================="
echo "Experiment:        $EXPERIMENT"
echo "Benchmark:         $BENCHMARK"
echo "Tasks folder:      $TASKS_FOLDER"
echo "Logs folder:       $LOGS_FOLDER"
echo "Output CSV:        $OUTPUT_CSV"
echo "Model:             $MODEL"
echo "==========================================="

# ---------------------------------------------------------
# Create folders if needed
# ---------------------------------------------------------
mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$LOGS_FOLDER"

# ---------------------------------------------------------
# Execute Python processing
# ---------------------------------------------------------
echo "Starting Python job..."
python "$PYTHON_SCRIPT" \
    --benchmark "$BENCHMARK" \
    --tasks "$TASKS_FOLDER" \
    --logs "$LOGS_FOLDER" \
    --output "$OUTPUT_CSV" \
    --model "$MODEL" \
    --experiment "$EXPERIMENT"

echo "All done!"
