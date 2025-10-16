#!/usr/bin/env python3
"""
Analyze benchmark dataset and generate both summary statistics
and disaggregated counts, including average word counts for prompt/context columns.

Example usage:
    python descriptives.py \
        --benchmark ../../data/benchmark/benchmark.csv \
        --difficulty "Difficulty: 1 (Easy) - 2 (Med) - 3 (Hard)" \
        --prompt "Prompt" \
        --context "Context/Background" \
        --output_dir ../../output/
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import re


# ----------------------------------------------------------------------
# Column mapping for benchmark dataset (adjust as needed)
# ----------------------------------------------------------------------
COLUMN_MAP = {
    "discovera": "Discovera (gpt-4o)",
    "llm": "LLM (gpt-4o)",
    "groundtruth": "Ground Truth",
    "biomni": "Biomni"
}


def analyze_benchmark_disaggregated(df, col_map, difficulty_col,
                                    prompt_col="Prompt", context_col="Context/Background"):
    """
    Analyze benchmark dataset and provide both summary stats and disaggregated counts.
    Computes average word count for prompt and context columns.

    Returns:
        stats_summary (pd.DataFrame): Dataset-level stats (total rows, unique papers, etc.)
        disaggregated_df (pd.DataFrame): All disaggregated counts combined into a single DataFrame.
    """
    df = df.copy()

    # --- Dataset-level stats ---
    total_rows = len(df)
    unique_papers = df["Paper Title"].nunique() if "Paper Title" in df.columns else None

    # Extract year if available
    if "Date Published" in df.columns:
        df["Year Published"] = pd.to_datetime(df["Date Published"], errors="coerce").dt.year
        unique_years = df["Year Published"].nunique()
        most_common_year = (
            df["Year Published"].mode().iloc[0] if df["Year Published"].notna().any() else None
        )
    else:
        unique_years = most_common_year = None

    unique_journals = df["Journal"].nunique() if "Journal" in df.columns else None
    unique_species = df["Species"].nunique() if "Species" in df.columns else None
    unique_difficulties = df[difficulty_col].nunique() if difficulty_col in df.columns else None
    unique_how_chosen = df["How Chosen"].nunique() if "How Chosen" in df.columns else None

    # --- Compute average word counts ---
    avg_prompt_wc = (
        df[prompt_col].astype(str).apply(lambda x: len(re.findall(r'\b\w+\b', x))).mean()
        if prompt_col in df.columns else None
    )
    avg_context_wc = (
        df[context_col].astype(str).apply(lambda x: len(re.findall(r'\b\w+\b', x))).mean()
        if context_col in df.columns else None
    )

    # --- Dataset stats summary ---
    stats_data = {
        "Metric": [
            "Total Tasks",
            "Unique Papers",
            "Unique Years",
            "Most Common Year",
            "Unique Journals",
            "Unique Species",
            "Unique Difficulty Levels",
            "Unique How it was Chosen",
            f"Avg Word Count - {prompt_col}",
            f"Avg Word Count - {context_col}"
        ],
        "Value": [
            total_rows,
            unique_papers,
            unique_years,
            most_common_year,
            unique_journals,
            unique_species,
            unique_difficulties,
            unique_how_chosen,
            round(avg_prompt_wc, 2) if avg_prompt_wc else None,
            round(avg_context_wc, 2) if avg_context_wc else None
        ]
    }
    stats_summary = pd.DataFrame(stats_data)

    # --- Disaggregated counts ---
    disaggregated_list = []

    def add_disagg(col_name, label, sort_index=False):
        if col_name in df.columns:
            temp = df[col_name].value_counts().sort_index() if sort_index else df[col_name].value_counts()
            temp = temp.reset_index()
            temp.columns = ["Category", "Count"]
            temp["Type"] = label
            disaggregated_list.append(temp)

    add_disagg("Year Published", "Year Published", sort_index=True)
    add_disagg("Journal", "Journal")
    add_disagg("Species", "Species")
    add_disagg(difficulty_col, "Difficulty", sort_index=True)
    add_disagg("How Chosen", "How Task Was Chosen?", sort_index=True)

    # Combine all into one DataFrame
    disaggregated_df = pd.concat(disaggregated_list, ignore_index=True)
    disaggregated_df = disaggregated_df[["Type", "Category", "Count"]]

    return stats_summary, disaggregated_df


# ----------------------------------------------------------------------
# CLI and logging setup
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze benchmark dataset and compute summary stats.")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to benchmark CSV file.")
    parser.add_argument("--difficulty", type=str, required=True, help="Name of difficulty column.")
    parser.add_argument("--prompt", type=str, default="Prompt", help="Prompt column name.")
    parser.add_argument("--context", type=str, default="Context/Background", help="Context column name.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../output/",
        help="Directory where results will be saved."
    )
    return parser.parse_args()


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")]
    )
    logging.info(f"Logging to {log_file}")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "benchmark_statistics.log"
    setup_logger(log_file)

    logging.info(f"Loading benchmark CSV: {args.benchmark}")
    df = pd.read_csv(args.benchmark)
    logging.info(f"Loaded {len(df)} rows.")

    stats_summary, disaggregated_df = analyze_benchmark_disaggregated(
        df,
        COLUMN_MAP,
        args.difficulty,
        prompt_col=args.prompt,
        context_col=args.context
    )

    # Output results
    stats_csv = output_path / "stats_bench_agg.csv"
    disagg_csv = output_path / "stats_bench_disa.csv"

    stats_summary.to_csv(stats_csv, index=False)
    disaggregated_df.to_csv(disagg_csv, index=False)

    logging.info("=== Dataset Statistics ===")
    logging.info(f"\n{stats_summary.to_string(index=False)}")

    logging.info("=== Disaggregated Counts ===")
    logging.info(f"\n{disaggregated_df.head(20).to_string(index=False)}")

    logging.info(f"Saved summary to {stats_csv}")
    logging.info(f"Saved disaggregated counts to {disagg_csv}")
    logging.info("Benchmark analysis completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
