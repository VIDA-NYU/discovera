"""
Script to generate answers based on benchmark reports.
Supports multiple report sources such as Ground Truth (GT), Discovera (gpt-4o), et
"""

import pandas as pd
import sys
import os
module_dir = "../../"  # adjust relative path
sys.path.append(os.path.abspath(module_dir))
from src.eval. prompting import *

# ---------------------------
# Configuration
# ---------------------------
GROUND_TRUTH = "groundtruth"  # Column name for GT reports
GENERATED_REPORT = "discovera(gpt-4o)"  # Source of generated reports
MODEL = "gpt-5"  # OpenAI model to use to answer questions
PROVIDER = "openai"  # Model provider used to answer questions
MAX_QUESTIONS = 10  # Maximum number of questions to process
BENCHMARK_PATH = "../data/benchmark/benchmark.csv"  # Path to benchmark CSV


def load_questions(max_questions: int, report_source: str, provider: str, model: str) -> list[dict]:
    """
    Load questions from JSON files and convert them to a list of dictionaries.

    Args:
        max_questions (int): Maximum number of questions to load.
        report_source (str): Report source identifier (GT or generated report).
        provider (str): Provider name (e.g., openai).
        model (str): Model name (e.g., gpt-5).

    Returns:
        list[dict]: List of questions as dictionaries.
    """
    file_path = f"../data/qs{max_questions}_rs{report_source}_{provider}_{model}.json"
    questions_df = pd.read_json(file_path)
    return questions_df.to_dict(orient="records")


def generate_responses(questions: list[dict], report_column: str, output_path: str, model: str) -> None:
    """
    Generate responses for questions using the specified report column.

    Args:
        questions (list[dict]): List of question dictionaries.
        report_column (str): Column name in benchmark CSV containing the reports.
        output_path (str): Path to save the generated responses.
        model (str): Model name to use for generating responses.
    """
    reports = load_reports(BENCHMARK_PATH, report_column=report_column)
    respond_question(
        questions=questions,
        reports=reports,
        output_path=output_path,
        model=model
    )
    
# ---------------------------
# Load questions
# ---------------------------

# Load questions generated GT and Discovera reports
questions_gt = load_questions(MAX_QUESTIONS, GROUND_TRUTH, PROVIDER, MODEL)
questions_discovera = load_questions(MAX_QUESTIONS, GENERATED_REPORT, PROVIDER, MODEL)

# Combine all questions
all_questions = questions_gt + questions_discovera


# ---------------------------
# Generate Answers
# ---------------------------

# Answers Using GT reports
generate_responses(all_questions, report_column="Ground Truth", output_path="../data/", model=MODEL)

# Answers Using Discovera reports
generate_responses(all_questions, report_column="Discovera (gpt-4o)", output_path="../data/", model=MODEL)