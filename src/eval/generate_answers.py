#!/usr/bin/env python3
"""
Generate pairwise answers: questions from a base report vs a compare report,
using report texts from benchmark CSV columns mapped automatically.

Questions are loaded from JSON files for both base and compare reports.

Example usage:
    python generate_answers.py \
        --experiment gpt-5_20250917_183930 \
        --benchmark ../../data/benchmark/benchmark.csv \
        --base groundtruth \
        --compare discovera(gpt-4o) \
        --model gpt-5
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd

# Add project root to Python path
module_dir = "../../"
sys.path.append(os.path.abspath(module_dir))

from src.eval.prompting import load_reports, respond_question  # noqa: E402


# Mapping JSON keywords to benchmark columns
REPORT_TO_COLUMN = {
    "discovera(gpt-4o)": "Discovera (gpt-4o)",
    "llm(gpt-4o)": "LLM (gpt-4o)",
    "groundtruth": "Ground Truth",
    "discovera(o4-mini)": "Discovera (o4-mini)",
    #"biomni": "Biomni",
    "biomni(11-05-25)": "Biomni (11-05-25)",
    "llm(o4-mini)": "LLM (o4-mini)",
    "biomni(o4-mini)": "Biomni (o4-mini)"

    # Add more mappings here if needed
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate answers from base and compare reports."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment folder containing questions."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Path to benchmark CSV file."
    )
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Substring to identify the base report JSON file."
    )
    parser.add_argument(
        "--compare",
        type=str,
        required=True,
        help="Substring to identify the compare report JSON file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model to use for generating answers."
    )
    return parser.parse_args()


def setup_logger(run_dir, model):
    log_file = os.path.join(run_dir, f"pairwise_answers_{model}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w")
        ]
    )
    logging.info(f"Logging to {log_file}")


def load_questions_from_reports(questions_folder: Path, report_keywords: list[str]) -> list[dict]:
    """
    Load questions from multiple JSON files identified by keywords.
    Returns a combined list of all questions.
    """
    all_questions = []
    for keyword in report_keywords:
        for file in questions_folder.glob("*.json"):
            if keyword in file.name:
                logging.info(f"Loading questions from {file.name}")
                df = pd.read_json(file)
                all_questions.extend(df.to_dict(orient="records"))
                break
        else:
            raise FileNotFoundError(f"No JSON file found with keyword '{keyword}' in {questions_folder}")
    return all_questions


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    base_column = REPORT_TO_COLUMN.get(args.base)
    print(f"Base column mapped: {base_column}")
    compare_column = REPORT_TO_COLUMN.get(args.compare)
    print(f"Compared column mapped: {compare_column}")

    if base_column is None or compare_column is None:
        raise ValueError(f"Could not find benchmark column mapping for base '{args.base}' or compare '{args.compare}'")

    base_path = Path("../../data/experiments") / args.experiment
    if not base_path.exists():
        raise FileNotFoundError(f"Experiment folder not found: {base_path}")

    answers_folder = base_path / "answers"
    answers_folder.mkdir(exist_ok=True)
    print(f"Answers will be saved to: {answers_folder}")
    setup_logger(answers_folder, args.model)

    questions_folder = base_path / "questions"

    # Efficiently load questions from base and compare JSONs
    report_keywords = [args.base, args.compare]
    # uncomment above and remove this soon, right now we are only checking how much of the ground truth questions are answered
    # but to follow the whole methodology we need both base and compare questions
    #report_keywords = [args.base]
    #report_keywords = [args.compare]

    all_questions = load_questions_from_reports(questions_folder, report_keywords)
    pair_folder = answers_folder / f"{args.base}_vs_{args.compare}"
    pair_folder.mkdir(exist_ok=True)

    # Generate answers using as context both types of reports: llm-generated and groundtruth
    columns_to_compare = [compare_column, base_column]

    for col in columns_to_compare:
        logging.info(f"Loading reports from benchmark column: {col}")
        reports = load_reports(args.benchmark, report_column=col)
        logging.info(f"Generating answers (model={args.model}, base={args.base}, compare={args.compare}, column={col})")
        respond_question(
            questions=all_questions,
            reports=reports,
            output_path=pair_folder,
            model=args.model
        )

    logging.info("Pairwise answer generation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


# """
# Script to generate answers based on benchmark reports.
# Supports multiple report sources such as Ground Truth (GT), Discovera (gpt-4o), et
# """

# import pandas as pd
# import sys
# import os
# module_dir = "../../"  # adjust relative path
# sys.path.append(os.path.abspath(module_dir))
# from src.eval. prompting import *

# # ---------------------------
# # Configuration
# # ---------------------------
# GROUND_TRUTH = "groundtruth"  # Column name for GT reports
# #GENERATED_REPORT = "discovera(gpt-4o)"  # Source of generated reports
# MODEL = "gpt-5"  # OpenAI model to use to answer questions
# PROVIDER = "openai"  # Model provider used to answer questions
# MAX_QUESTIONS = 10  # Maximum number of questions to process
# BENCHMARK_PATH = "../../data/benchmark/benchmark.csv"  # Path to benchmark CSV


# def load_questions(max_questions: int, report_source: str, provider: str, model: str) -> list[dict]:
#     """
#     Load questions from JSON files and convert them to a list of dictionaries.

#     Args:
#         max_questions (int): Maximum number of questions to load.
#         report_source (str): Report source identifier (GT or generated report).
#         provider (str): Provider name (e.g., openai).
#         model (str): Model name (e.g., gpt-5).

#     Returns:
#         list[dict]: List of questions as dictionaries.
#     """
#     file_path = f"../../data/questions/qs{max_questions}_rs{report_source}_{provider}_{model}.json"
#     questions_df = pd.read_json(file_path)
#     return questions_df.to_dict(orient="records")


# def generate_responses(questions: list[dict], report_column: str, output_path: str, model: str) -> None:
#     """
#     Generate responses for questions using the specified report column.

#     Args:
#         questions (list[dict]): List of question dictionaries.
#         report_column (str): Column name in benchmark CSV containing the reports.
#         output_path (str): Path to save the generated responses.
#         model (str): Model name to use for generating responses.
#     """
#     reports = load_reports(BENCHMARK_PATH, report_column=report_column)
#     respond_question(
#         questions=questions,
#         reports=reports,
#         output_path=output_path,
#         model=model
#     )
    
# # ---------------------------
# # Load questions
# # ---------------------------

# # Load questions generated GT and Discovera reports
# questions_gt = load_questions(MAX_QUESTIONS, GROUND_TRUTH, PROVIDER, MODEL)
# #questions_discovera = load_questions(MAX_QUESTIONS, GENERATED_REPORT, PROVIDER, MODEL)

# # Combine all questions
# #all_questions = questions_gt + questions_discovera


# # ---------------------------
# # Generate Answers GT AND DISCOVERA
# # ---------------------------

# # Answers Using GT reports
# #generate_responses(all_questions, report_column="Ground Truth", output_path="../../data/answers/", model=MODEL)
# #
# ## Answers Using Discovera reports
# #generate_responses(all_questions, report_column="Discovera (gpt-4o)", output_path="../../data/answers/", model=MODEL)

# # ---------------------------
# # Generate Answers GT AND LLM
# # ---------------------------

# # Load questions generated GT and Discovera reports
# GENERATED_REPORT = "llm(gpt-4o)"  # Source of generated reports
# #
# questions_llm = load_questions(MAX_QUESTIONS, GENERATED_REPORT, PROVIDER, MODEL)
# #
# ## Combine all questions
# all_questions = questions_gt + questions_llm
# #
# #
# ## ---------------------------
# ## Generate Answers
# ## ---------------------------
# #
# ## Answers Using GT reports
# generate_responses(all_questions, report_column="Ground Truth", output_path="../data/answers/", model=MODEL)

# # Answers Using LLM reports
# generate_responses(all_questions, report_column="LLM (gpt-4o)", output_path="../data/answers/", model=MODEL)