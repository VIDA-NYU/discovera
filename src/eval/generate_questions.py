#!/usr/bin/env python3
"""
Generate questions based on content from Ground Truth (GT), Discovera or LLM-generated reports.

Each run is logged to a timestamped log file and outputs are saved
with unique prefixes to avoid overwriting previous results.

Example usage:
    python generate_questions.py \
        --benchmark ../data/benchmark/benchmark.csv \
        --report-columns "Ground Truth" "Discovera (gpt-4o)" \
        --ns 10 \
        --model gpt-5 \
        --out ../data/questions/
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to Python path
module_dir = "../../"
sys.path.append(os.path.abspath(module_dir))

from src.eval.prompting import load_reports, generate_questions, multiple_questions_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate questions from GT or LLM reports."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark CSV file."
    )
    parser.add_argument(
        "--report-columns",
        nargs="+",
        required=True,
        help="Column names in benchmark CSV containing report text."
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=10,
        help="Max number of questions to generate per report."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model to use to generate questions."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../data/questions/",
        help="Output directory for generated questions."
    )
    return parser.parse_args()


def setup_logger(out_dir, model):
    """Configure logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"run_{model}_{timestamp}.log")

    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w")
        ]
    )
    logging.info(f"Logging to {log_file}")


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    setup_logger(args.out, args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for col in args.report_columns:
        logging.info(f"Loading reports from column: {col}")
        reports = load_reports(args.benchmark, report_column=col)

        prefix = f"{args.model}_{col.replace(' ', '_')}_{timestamp}"
        col_out = os.path.join(args.out, prefix)
        os.makedirs(col_out, exist_ok=True)

        logging.info(
            f"Generating {args.ns} questions per report "
            f"(model={args.model}, out={col_out})"
        )
        generate_questions(
            reports=reports,
            prompt_template=multiple_questions_template,
            model=args.model,
            num_questions=args.ns,
            output_path=col_out
        )

    logging.info("Question generation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
