#!/usr/bin/env python3
"""
Script to generate questions from both ground truth (GT) reports
and LLM-generated reports using a specified model and template.

Workflow:
1. Load reports (GT and LLM-generated).
2. Generate a specified number of questions from each report set.
3. Save the generated questions to the output directory.

Configuration:
- Adjust `ns` to control number of questions per report.
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
ns = 10                  # number of questions to generate per report
out = "../data/"         # output directory for generated questions
model = "gpt-5"          # model used for question generation

# ---------------------------
# Ground Truth Reports
# ---------------------------
src = "gt"  # label for ground truth source
reports_gt = load_reports(
    "../data/benchmark/benchmark.csv",  # path to GT reports
    report_column="Ground Truth",       # column containing report text
    source=src
)

questions_gt_report = generate_questions(
    reports=reports_gt,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out,
    source=src
)

# ---------------------------
# LLM-Generated Reports
# ---------------------------
src = "llm"  # label for LLM-generated source
reports_llm = load_reports(
    "../data/llm_generated_reports_gpt-5.csv",  # path to LLM generated reports
    report_column="LLM Report Column",          # column containing report
    source=src
)

questions_llm_generated_report = generate_questions(
    reports=reports_llm,
    prompt_template=multiple_questions_template,
    model=model,
    num_questions=ns,
    output_path=out,
    source=src
)

print("Question generation completed for both GT and LLM reports.")
