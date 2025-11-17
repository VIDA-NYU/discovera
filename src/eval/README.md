# Evaluation Guide

This guide explains how to use the evaluation framework to generate questions, generate answers, and visualize results.

## Prerequisites

1. Set up your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Ensure you have the required dependencies installed (pandas, matplotlib, seaborn, tqdm, etc.)

## Overview

The evaluation workflow consists of three main steps:

1. **Generate Questions**: Create multiple-choice questions from benchmark breakdown CSV files
2. **Generate Answers**: Answer questions using different report sources (Discovera, LLM, Biomni, etc.)
3. **Visualize Results**: Create comparison tables and visualizations to analyze performance

---

## Step 1: Generate Questions

Generate multiple-choice questions from a benchmark breakdown CSV file. Questions are generated based on keypoints and context for each task.

### Function: `generate_questions_from_breakdown`

**Location**: `eval.prompting`

**Parameters**:
- `csv_path` (str): Path to the benchmark breakdown CSV file
- `provider` (str, optional): LLM provider (e.g., "openai", "anthropic"). Default: "openai"
- `model` (str, optional): Model name to use for question generation. Default: "gpt-4o"
- `output_path` (str, optional): Directory to save generated questions. If None, questions are returned but not saved
- `task_filter` (List[str], optional): List of task IDs to process. If None, processes all tasks

**Example Usage**:

```python
from eval.prompting import generate_questions_from_breakdown

questions = generate_questions_from_breakdown(
    csv_path="benchmark_breakdown_new.csv",
    provider="openai",
    model="gpt-5-nano",
    output_path="output/questions_new_1125",
    task_filter=["9.1", "9.2", "9.3", "9.4", "9.5", "10", "11.1", "11.2", "11.3", "12"],
)
```

**Output Structure**:
- Questions are saved in JSON files organized by task ID
- Each task directory contains a file named: `qs{num}_rsgroundtruth_{provider}_{model}.json`
- Each question includes:
  - `task_id`: Task identifier
  - `question_id`: Keypoint number
  - `question`: The multiple-choice question text
  - `choices`: List of answer choices (A, B, C, D)
  - `answer`: Correct answer (A, B, C, or D)
  - `ground_truth_keypoint`: Original keypoint text
  - `fine_grained_prompt`: Associated prompt

**Notes**:
- Questions are generated in batches by task (one batch per task)
- One question is generated per keypoint in the breakdown CSV
- The function filters keypoints based on the `task_filter` parameter

---

## Step 2: Generate Answers

Answer the generated questions using reports from different sources (Discovera, LLM, Biomni, etc.).

### Function: `respond_questions_from_breakdown`

**Location**: `eval.prompting`

**Parameters**:
- `task_ids` (List[str]): List of task IDs to process (e.g., `["1", "2.1", "9.1"]`)
- `csv_path` (str): Path to the benchmark CSV file containing reports
- `questions_dir` (str): Directory containing question JSON files (e.g., `"output/questions_new_1125"`)
- `answers_dir` (str): Directory to save answer files (e.g., `"output/answers_new_1125"`)
- `report_columns` (List[str]): List of CSV column names to use as reports (e.g., `["Discovera (o4-mini)", "LLM (o4-mini)", "Biomni (o4-mini)"]`)
- `provider` (str, optional): LLM provider. Default: "openai"
- `model` (str, optional): Model name for answering. Default: "gpt-4o"

**Example Usage**:

```python
from eval.prompting import respond_questions_from_breakdown

task_ids = ["9.1", "9.2", "9.3", "9.4", "9.5", "10", "11.1", "11.2", "11.3", "12"]

respond_questions_from_breakdown(
    task_ids=task_ids,
    csv_path="benchmark_verified.csv",
    questions_dir="output/questions_new_1125",
    answers_dir="output/answers_new_1125",
    report_columns=[
        "Discovera (claude-3.5-haiku)",
        "LLM (claude-3.5-haiku)",
        "Biomni (claude-3.5-haiku)",
    ],
    provider="openai",
    model="gpt-5-nano",
)
```

**Output Structure**:
- Answers are saved in JSON files organized by task ID
- Each task directory contains files named: `ans{num}_rs{source}_{provider}_{model}.json`
- Each answer includes:
  - `task_id`: Task identifier
  - `question_id`: Question identifier
  - `question`: Question text
  - `choices`: Answer choices
  - `answer`: Correct answer
  - `prediction`: Model's predicted answer
  - `confidence`: Model's confidence score (0-1)
  - `report_source`: Source of the report used (normalized column name)

**Notes**:
- The function processes all question files found in each task directory
- For each report column specified, it generates a separate answer file
- Report column names are normalized (lowercased, spaces removed) for file naming

### Alternative: Fine-grained Reports

If you have fine-grained reports stored in a directory structure, you can use:

```python
from eval.prompting import respond_questions_with_finegrained_reports

respond_questions_with_finegrained_reports(
    questions_dir="output/questions_new_1118",
    reports_dir="/path/to/reports/Discovera",
    answers_dir="output/answers_new",
    provider="openai",
    model="gpt-5-nano",
    task_ids=["11.3"],  # Optional: filter specific tasks
)
```

---

## Step 3: Visualize Results

Generate comparison tables and visualizations to analyze the performance of different report sources.

### Function: `generate_score_comparison_table`

**Location**: `eval.prompting`

**Parameters**:
- `answers_dir` (str): Directory containing answer JSON files (e.g., `"output/answers_new_1125/"`)
- `task_ids` (List[str], optional): List of task IDs to process. If None, processes all tasks found in the directory

**Returns**: `pandas.DataFrame` with columns:
- `task_id`: Task identifier
- `report_source`: Report source (e.g., "discovera(o4-mini)")
- `provider`: LLM provider
- `model`: Model name
- `total_questions`: Total number of questions
- `correct`: Number of correct answers
- `accuracy`: Accuracy score (0-1)
- `avg_confidence`: Average confidence score
- `filename`: Source filename

**Example Usage**:

```python
from eval.prompting import generate_score_comparison_table
import pandas as pd

task_ids = [
    "1", "2.1", "3.1", "3.2", "3.3", "4", "5.1", "5.2", "6",
    "7.1", "7.2", "8", "9.1", "9.2", "9.3", "9.4", "9.5",
    "10", "11.1", "11.2", "11.3", "12",
]

# Generate comparison table
df = generate_score_comparison_table("output/answers_new_1125/", task_ids=task_ids)

# Save to CSV
df.to_csv("results.csv")

# Create pivot table for easier comparison
pivot = df.pivot_table(
    index="task_id",
    columns="report_source",
    values="accuracy"
)
```

### Visualization Example

Create visualizations to compare performance across methods and tasks:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

# Filter methods of interest
methods = [
    "llm(o4-mini)",
    "biomni(o4-mini)",
    "discovera(o4-mini)",
    "llm(claude-3.5-haiku)",
    "biomni(claude-3.5-haiku)",
    "discovera(claude-3.5-haiku)",
]

pivot_filtered = pivot[methods]

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 1. Grouped Bar Chart
ax1 = axes[0]
pivot_filtered.plot(
    kind="bar", ax=ax1, width=0.8,
    color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
)
ax1.set_title("Accuracy Comparison Across Methods by Task", fontsize=14, fontweight="bold", pad=20)
ax1.set_xlabel("Task ID", fontsize=12, fontweight="bold")
ax1.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax1.set_ylim([0, 1.1])
ax1.legend(title="Report Source", title_fontsize=11, fontsize=10, loc="upper left", frameon=True)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha="center")
ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="50% baseline")

# Add value labels on bars
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8, padding=2)

# 2. Heatmap
ax2 = axes[1]
pivot_sorted = pivot_filtered.sort_index()
sns.heatmap(
    pivot_sorted.T,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    center=0.5,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Accuracy"},
    ax=ax2,
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"size": 9, "weight": "bold"},
)
ax2.set_title("Accuracy Heatmap: Methods vs Tasks", fontsize=14, fontweight="bold", pad=20)
ax2.set_xlabel("Task ID", fontsize=12, fontweight="bold")
ax2.set_ylabel("Report Source", fontsize=12, fontweight="bold")
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"\nOverall Average Accuracy by Tasks:")
print(pivot_filtered.mean().sort_values(ascending=False))

# Compute weighted average accuracy
print(f"\nOverall Average Accuracy by Questions:")
weighted_avg = {}
for method in methods:
    method_df = df[df["report_source"] == method]
    if len(method_df) > 0:
        numerator = method_df["correct"].sum()
        denominator = method_df["total_questions"].sum()
        weighted_avg[method] = numerator / denominator if denominator > 0 else 0.0
    else:
        weighted_avg[method] = 0.0

weighted_series = pd.Series(weighted_avg).sort_values(ascending=False)
print(weighted_series)
```

---

## Complete Workflow Example

Here's a complete example that ties everything together:

```python
from eval.prompting import (
    generate_questions_from_breakdown,
    respond_questions_from_breakdown,
    generate_score_comparison_table,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate questions
questions = generate_questions_from_breakdown(
    csv_path="benchmark_breakdown_new.csv",
    provider="openai",
    model="gpt-5-nano",
    output_path="output/questions_new_1125",
    task_filter=["9.1", "9.2", "9.3", "10"],
)

# Step 2: Generate answers
task_ids = ["9.1", "9.2", "9.3", "10"]
respond_questions_from_breakdown(
    task_ids=task_ids,
    csv_path="benchmark_verified.csv",
    questions_dir="output/questions_new_1125",
    answers_dir="output/answers_new_1125",
    report_columns=[
        "Discovera (claude-3.5-haiku)",
        "LLM (claude-3.5-haiku)",
        "Biomni (claude-3.5-haiku)",
    ],
    provider="openai",
    model="gpt-5-nano",
)

# Step 3: Generate comparison table
df = generate_score_comparison_table("output/answers_new_1125/", task_ids=task_ids)
df.to_csv("results.csv")

# Step 4: Create pivot table and visualize
pivot = df.pivot_table(index="task_id", columns="report_source", values="accuracy")
print(pivot)
```

---

## File Structure

The evaluation framework expects the following directory structure:

```
output/
├── questions_new_1125/
│   ├── 9.1/
│   │   └── qs7_rsgroundtruth_openai_gpt_5_nano.json
│   ├── 9.2/
│   │   └── qs6_rsgroundtruth_openai_gpt_5_nano.json
│   └── ...
└── answers_new_1125/
    ├── 9.1/
    │   ├── ans7_rsdiscovera(claude-3.5-haiku)_openai_gpt_5_nano.json
    │   ├── ans7_rsllm(claude-3.5-haiku)_openai_gpt_5_nano.json
    │   └── ans7_rsbiomni(claude-3.5-haiku)_openai_gpt_5_nano.json
    └── ...
```

---

## Tips and Best Practices

1. **Task Filtering**: Use `task_filter` when generating questions to focus on specific tasks and reduce processing time.

2. **Model Selection**: Choose appropriate models for question generation vs. answering:
   - Question generation: Use more capable models (e.g., `gpt-5-nano`, `gpt-4o`) for better question quality
   - Answering: Can use faster/cheaper models for evaluation

3. **Report Columns**: Ensure CSV column names match exactly (including spaces and capitalization) when specifying `report_columns`.

4. **Error Handling**: The functions include error handling and will continue processing even if individual tasks fail. Check logs for warnings.

5. **Weighted Accuracy**: When comparing methods, consider both task-level and question-level weighted averages, as tasks may have different numbers of questions.

6. **Visualization**: Customize visualizations based on your needs. The heatmap is particularly useful for identifying patterns across tasks and methods.

---

## Troubleshooting

**Issue**: Questions not generating
- Check that the CSV file exists and has the correct format
- Verify that task IDs in `task_filter` match those in the CSV
- Check API key is set correctly

**Issue**: Answers not generating
- Ensure question files exist in the specified `questions_dir`
- Verify CSV file contains the specified `report_columns`
- Check that task IDs match between questions and CSV

**Issue**: Empty comparison table
- Verify answer files exist and are in the correct format
- Check that filenames follow the expected pattern: `ans{num}_rs{source}_{provider}_{model}.json`
- Ensure task IDs match between answers directory and `task_ids` parameter

