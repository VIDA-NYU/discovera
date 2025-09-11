#!/usr/bin/env python3
"""
Script to generate questions from both ground truth (GT) reports
and LLM-generated reports using a specified model and template.

Workflow:
1. Load reports (GT, Discovera, and LLM-generated).
2. Generate a specified maximum number of questions from each report set.
3. Save the generated questions to the output directory.

Configuration:
- Adjust `ns` to control the max number of questions per report.
- Update `out` to set output directory.
- Change `model` to select which LLM model is used.
"""

import sys
import os

# Add project root to Python path so local modules can be imported
module_dir = "../"  # relative path to project root (adjust if needed)
sys.path.append(os.path.abspath(module_dir))

# Import required functions and templates from evaluation module
from src.eval.prompting import load_reports, generate_questions, multiple_questions_template

# ---------------------------
# Configuration Parameters
# ---------------------------
ns = 10                  # max number of questions to generate per report
out = "../data/"         # output directory for generated questions
model = "gpt-5"          # model used for question generation
benchmark_data_path = "../data/benchmark/benchmark.csv"  

# ---------------------------
# Ground Truth Reports
# ---------------------------
reports_gt = load_reports(
    benchmark_data_path,  # path to GT reports
    report_column="Ground Truth",       # column containing report text
)

questions_gt_report = generate_questions(
    reports=reports_gt,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out
)

# ---------------------------
# Discovera-Generated or Simply LLM-Generated Reports
# ---------------------------

# Discovera generated reports that were created used using gpt-4o
reports_discovera_a = load_reports(
    benchmark_data_path,  
    report_column="Discovera (gpt-4o)", # column containing report
)

generate_questions(
    reports=reports_discovera_a,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out
    )

# Load Discovera generated reports that were created used using gpt-5-nano

reports_discovera_b = load_reports(
    benchmark_data_path,  
    report_column="Discovera (gpt-5-nano)", # column containing report
)

generate_questions(
    reports=reports_discovera_b,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out
    )

# Load LLM generated reports that were created used using gpt-5

reports_llm = load_reports(
    benchmark_data_path,  
    report_column="LLM (gpt-5)", # column containing report
)

generate_questions(
    reports=reports_llm,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out
    )

print("Question generation completed for both GT and LLM reports.")
