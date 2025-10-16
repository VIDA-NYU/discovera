#!/usr/bin/env python3
"""
Purpose:
    Compute variable-level and overall agreement summaries for multiple report types.

Usage:
    python compile_results.py \
        --benchmark_path PATH_TO_BENCHMARK \
        --experiment_id EXPERIMENT_ID \
        --base_dir BASE_DIRECTORY \
        --report_types llm(gpt-4o) discovera(gpt-4o)
"""

import argparse
from pathlib import Path
import sys
import os
# -----------------------------
# Adjust module path
# -----------------------------
module_dir = "../../"  # relative path to src folder
sys.path.append(os.path.abspath(module_dir))

# -----------------------------
# Import functions
# -----------------------------

from src.eval.evaluation import compute_all_reports

def parse_args():
    parser = argparse.ArgumentParser(description="Run agreement summary analysis for multiple report types.")
    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to benchmark CSV file"
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID (folder name under experiments/)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing experiments/"
    )
    parser.add_argument(
        "--report_types",
        nargs="+",
        required=True,
        help="List of report types to process, e.g., llm(gpt-4o) discovera(gpt-4o)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    compute_all_reports(
        benchmark_path=args.benchmark_path,
        experiment_id=args.experiment_id,
        base_dir=args.base_dir,
        report_types=args.report_types
    )

    print(f"Completed processing {len(args.report_types)} report types.")


if __name__ == "__main__":
    main()
