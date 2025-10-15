#!/usr/bin/env python3
"""
Purpose:
    Compute semantic textual similarity (STS) and ROUGE-L between model-generated
    text columns and a specified ground truth column.

Inputs:
    - CSV or Excel file with text columns
    - Ground truth column name
    - One or more comparison column names

Outputs:
    - CSV with similarity metrics
    - Summary CSV
    - Logs in output directory
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

# -----------------------------
# Adjust module path
# -----------------------------
module_dir = "../../"  # relative path to src folder
sys.path.append(os.path.abspath(module_dir))

from src.eval.evaluation import traditional_similarity_metrics

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Compute STS and ROUGE similarity scores.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV or Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../output/",
        help="Directory where results will be saved."
    )
    parser.add_argument(
        "--sts_model",
        type=str,
        default="dmlls/all-mpnet-base-v2-negation",
        help="SentenceTransformer model name."
    )
    parser.add_argument(
        "--ground_truth_col",
        type=str,
        required=True,
        help="Column name to use as ground truth."
    )
    parser.add_argument(
        "--compare_cols",
        nargs="+",
        required=True,
        help="List of column names to compare against the ground truth."
    )
    return parser.parse_args()


# -----------------------------
# Logger setup
# -----------------------------
def setup_logger(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "similarity_analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")]
    )
    logging.info(f"Logging to {log_file}")


# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    args = parse_args()
    setup_logger(Path(args.output_dir))
    traditional_similarity_metrics(
        input_path=args.input_path,
        output_dir=args.output_dir,
        sts_model=args.sts_model,
        ground_truth_col=args.ground_truth_col,
        compare_cols=args.compare_cols
    )


if __name__ == "__main__":
    main()
