#!/usr/bin/env python3
"""
Generate questions based on content from Ground Truth (GT), Discovera or LLM-generated reports.

Each run is saved into a timestamped folder. Logs and outputs are stored
inside that folder to keep runs organized and avoid overwriting results.

Example usage:
    python generate_questions.py \
        --benchmark ../../data/benchmark/benchmark.csv \
        --report-columns "Ground Truth" "Discovera (gpt-4o)" \
        --ns 10 \
        --model gpt-5 \
        --out ../../data/experiments/
"""

import sys
import os
import argparse
import logging
import inspect

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
        help="Base output directory for generated questions."
    )
    return parser.parse_args()


def setup_logger(run_dir, model):
    """Configure logging to both console and file."""
    log_file = os.path.join(run_dir, f"run_{model}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w")
        ]
    )
    logging.info(f"Logging to {log_file}")


def save_prompt_template(run_dir, prompt_template):
    """Save the prompt template function source for reproducibility."""
    template_file = os.path.join(run_dir, "prompt_template.txt")
    with open(template_file, "w", encoding="utf-8") as f:
        f.write(inspect.getsource(prompt_template))
    logging.info(f"Prompt template source saved to {template_file}")

def main():
    args = parse_args()

    # Fixed timestamp folder for the whole run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.out, f"{args.model}_{timestamp}/questions")
    os.makedirs(experiment_dir, exist_ok=True)

    setup_logger(experiment_dir, args.model)

    # Save the current prompt template for reproducibility
    save_prompt_template(experiment_dir, multiple_questions_template)

    for col in args.report_columns:
        logging.info(f"Loading reports from column: {col}")
        reports = load_reports(args.benchmark, report_column=col)

        logging.info(
            f"Generating max {args.ns} questions per report "
            f"(model={args.model}, out={experiment_dir})"
        )
        generate_questions(
            reports=reports,
            prompt_template=multiple_questions_template,
            model=args.model,
            num_questions=args.ns,
            output_path=experiment_dir
        )

    logging.info("Question generation completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)