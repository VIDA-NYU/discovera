# #!/usr/bin/env bash
# # Bash wrapper to run generate_questions.py with multiple settings

# set -euo pipefail

# # ------------------------
# # Config (edit as needed)
# # ------------------------
# BENCHMARK="../../data/benchmark/benchmark.csv"
# OUTDIR="../../data/experiments/"
# NS=1 # Number of samples per question

# # List of models to try
# MODELS=("gpt-5")

# # List of report columns
# REPORT_COLUMNS=(
#   #"LLM (gpt-4o)" 
#   #"LLM (o4-mini)"
#   #"Discovera (gpt-4o)" 
#   #"Discovera (o4-mini)"
#   #"Ground Truth"
#   #"Biomni (claude-3.5-haiku)"
#   "Biomni (o4-mini)"
#   )

# # ------------------------
# # Run
# # ------------------------
# for MODEL in "${MODELS[@]}"; do
#   echo "=== Running model: $MODEL ==="
  
#   # Join report columns with quotes for Python
#   COLS=()
#   for COL in "${REPORT_COLUMNS[@]}"; do
#     COLS+=("$COL")
#   done
  
#   python3 generate_questions.py \
#     --benchmark "$BENCHMARK" \
#     --report-columns "${COLS[@]}" \
#     --ns "$NS" \
#     --model "$MODEL" \
#     --out "$OUTDIR"
  
# done


#!/usr/bin/env bash
# Bash wrapper to run generate_questions.py with multiple settings

set -euo pipefail

# ------------------------
# Config (edit as needed)
# ------------------------
BENCHMARK="../../data/benchmark/benchmark.csv"
OUTDIR="../../data/experiments/"
NS=1 # Number of samples per question

# List of models to try
MODELS=("gpt-5")

# List of report columns
REPORT_COLUMNS=(
  "Ground Truth"
  "Biomni (o4-mini)"
)

# Optional: comma-separated TASK_IDS to run, e.g., "1,5,10"
#TASK_IDS=""
TASK_IDS="1,2.1,10"

# ------------------------
# Run
# ------------------------
for MODEL in "${MODELS[@]}"; do
  echo "=== Running model: $MODEL ==="

  # Generate timestamp for this run
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  EXPERIMENT_ID="${MODEL}_${TIMESTAMP}"

  echo "Experiment ID: $EXPERIMENT_ID"  # <- master script will read this

  # Join report columns with quotes for Python
  COLS=()
  for COL in "${REPORT_COLUMNS[@]}"; do
    COLS+=("$COL")
  done

  ARGS=(
    --benchmark "$BENCHMARK"
    --report-columns "${COLS[@]}"
    --ns "$NS"
    --model "$MODEL"
    --out "$OUTDIR"
    --experiment-id "$EXPERIMENT_ID"
  )

  # Add optional task IDs
  if [ -n "$TASK_IDS" ]; then
    IFS=',' read -ra IDS_ARRAY <<< "$TASK_IDS"
    ARGS+=(--task-ids "${IDS_ARRAY[@]}")
  fi

  # --- Print parameters before running ---
  echo "[INFO] Running generate_questions.py with parameters:"
  echo "       Benchmark: $BENCHMARK"
  echo "       Report columns: ${COLS[*]}"
  echo "       Maximum number of questions per task: $NS"
  echo "       Model: $MODEL"
  echo "       Output dir: $OUTDIR"
  if [ -n "$TASK_IDS" ]; then
    echo "       Task IDs: ${IDS_ARRAY[*]}"
  else
    echo "       Task IDs: ALL"
  fi
  echo ""

  python3 generate_questions.py "${ARGS[@]}"
done
