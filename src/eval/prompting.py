import time
import os
import toml 
import json
import re
import csv

from typing import List, Callable, Optional, Union, Dict, Any
from tqdm import tqdm
from collections import Counter
from statistics import mode, StatisticsError


def load_openai_key(beaker_conf_path="../.beaker.conf"):
    """
    Load OpenAI API key from a Beaker configuration file.

    Args:
        beaker_conf_path (str): Path to the .beaker.conf file.

    Returns:
        str: OpenAI API key.
    """
    config = toml.load(beaker_conf_path)
    print("Beaker config loaded successfully.")

    openai_api_key = config['providers']['openai']['api_key']
    #print("Loaded API key (first 10 chars):", openai_api_key[:10] + "...")
    
    return openai_api_key

def load_reports(csv_path: str, report_column: str = "Ground Truth", id_column: str = "ID", source: str = "gt"):
    reports = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            report_id = row.get(id_column)
            report_text = row.get(report_column)
            if report_text and report_text.strip():
                report_key = f"{source}_{idx}"
                reports.append((report_key, report_id, report_text))
    return reports


class OpenAILLM:
    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def run(self, prompt: str, json_output: bool = True):
        """
        Sends a prompt to the model and returns the response.
        
        Parameters:
        - prompt: str, the instruction or task for the model.
        - json_output: bool, if True, parse output as JSON, else return raw text.
        
        Returns:
        - dict or str: Parsed JSON if json_output=True, else raw text.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        content = resp.choices[0].message.content.strip()
        if json_output:
            try:
                return json.loads(content)
            except Exception:
                content_clean = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.DOTALL).strip()
                return json.loads(content_clean)

def get_llm(backend: str, model: str = None, **kwargs):
    """
    Initialize an LLM client based on the selected backend and model.

    Args:
        backend: The LLM provider name (e.g., "openai").
        model: The model name to use (e.g., "gpt-4o-mini").
        kwargs: Additional arguments like API keys.

    Returns:
        An instance of an LLM wrapper class with a .run() method.
    """
    if backend == "openai":
        from openai import OpenAI
        api_key = kwargs.get("api_key")
        print(f"[INFO] Initializing LLM: backend='{backend}', model='{model}'")

        return OpenAILLM(OpenAI(api_key=api_key), model=model)

    elif backend == "anthropic":
        # TODO: Add Anthropic support
        pass

    elif backend == "hf":
        # TODO: Add HuggingFace support
        pass

    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

def multiple_questions_template(report_text: str, n: int) -> str:
    return f"""
    You are a biomedical researcher. Your task is to generate {n} multiple-choice questions based strictly on the report below.
    
    --- BEGIN REPORT ---
    {report_text}
    --- END REPORT ---
    
    Instructions:
    - The questions must be fully answerable using ONLY the content of the report above.
    - Do NOT use outside knowledge or assumptions.
    - Each question should have exactly five answer choices labeled as follows:
      A. <choice text>
      B. <choice text>
      C. <choice text>
      D. <choice text>
      E. Insufficient information
    - The choices themselves should be strings starting with the letter and period, e.g. "A. Pathway A".
    - Provide the correct answer as a single uppercase letter: "A", "B", "C", "D", or "E".
    
    Output format:
    A JSON list of {n} objects, where each object contains:
    - "question": string,
    - "choices": list of 5 strings (each starting with "A. ", "B. ", etc., last one is "E. Insufficient information"),
    - "correct": string (one of "A", "B", "C", "D", or "E")
    
    Example output:
    [
      {{
        "question": "What pathway was most enriched in the analysis?",
        "choices": [
          "A. Pathway A",
          "B. Pathway B",
          "C. Pathway C",
          "D. Pathway D",
          "E. Insufficient information"
        ],
        "correct": "A"
      }},
      {{
        "question": "What gene was mutated?",
        "choices": [
          "A. Gene X",
          "B. Gene Y",
          "C. Gene Z",
          "D. Gene W",
          "E. Insufficient information"
        ],
        "correct": "E"
      }},
      ...
    ]
    """


def answer_prompt_template(
    question: str, choices: List[str], report: str
) -> str:
    if report:
        prompt = f"""
        You are a biomedical domain expert. Use ONLY the information provided in the report below to answer the question.

        --- BEGIN REPORT ---
        {report}
        --- END REPORT ---

        Question:
        {question}

        Choices:
        A. {choices[0]}
        B. {choices[1]}
        C. {choices[2]}
        D. {choices[3]}
        E. {choices[4]}

        Instructions:
        - Select the best choice ONLY if it is explicitly stated or can be directly inferred from the report.
        - Do NOT use any outside knowledge, assumptions, or speculation.
        - If the answer cannot be determined from the report, respond with "E".
        
        Respond ONLY in this JSON format:
        {{
          "answer": "<A/B/C/D/E>",
          "confidence": <float between 0 and 1>
        }}
        """.strip()
    else:
        prompt = f"""
        You are a biomedical domain expert.

        Question:
        {question}

        Choices:
        A. {choices[0]}
        B. {choices[1]}
        C. {choices[2]}
        D. {choices[3]}
        E. {choices[4]}

        Instructions:
        - Answer the question based on your biomedical knowledge.
        - Provide the best choice letter ("A", "B", "C", or "D") and a confidence score between 0 and 1.
        - If you cannot confidently answer, respond with "E".

        Respond ONLY in this JSON format:
        {{
          "answer": "<A/B/C/D/E>",
          "confidence": <float between 0 and 1>
        }}
        """.strip()

    return prompt


def generate_questions(
    reports: List[tuple],
    prompt_template: Callable[[str, int], str],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    num_questions: Union[int, Dict[str, int]] = 20,
    output_path: Optional[str] = None,
    source: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate MCQs per report, allowing a fixed count or custom count per report.

    Args:
        num_questions: Either an int (same for all) or a dict mapping report_id → count.
    """
    llm = get_llm(provider, model, api_key=load_openai_key())
    start_time = time.time()

    print(f"[INFO] Generating questions for {len(reports)} reports...")
    all_questions = []

    # Determine if we're using custom counts
    custom_counts = isinstance(num_questions, dict)

    for report in tqdm(reports, desc="Processing Reports"):
        report_id = report[0]
        report_text = report[2] if len(report) > 2 else report[1]
        
        # Decide per-report question count
        if custom_counts:
            count = num_questions.get(report_id)
            if count is None:
                print(f"[WARNING] No num_questions specified for {report_id}. Skipping.")
                continue
        else:
            count = num_questions  # single fixed integer

        prompt = prompt_template(report_text, count)

        try:
            mcqs = llm.run(prompt, json_output=True)
            if isinstance(mcqs, list):
                for q in mcqs:
                    all_questions.append({
                        "report_id": report_id,
                        "question": q.get("question", ""),
                        "choices": q.get("choices", []),
                        "answer": q.get("correct", ""),
                        "question_source": source
                    })
            else:
                print(f"[WARNING] Unexpected format for {report_id}")
        except Exception as e:
            print(f"[ERROR] Report {report_id} failed: {e}")

    elapsed = time.time() - start_time
    print(f"[INFO] Completed in {elapsed:.2f} seconds. Total questions: {len(all_questions)}")


    if output_path:
        os.makedirs(output_path, exist_ok=True)
        suffix = (
            f"{num_questions}" if not custom_counts
            else f"cus"
        )
        filename = f"qs{suffix}_{source}_{provider}_{model.replace('/', '-')}.json"
        file_path = os.path.join(output_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved to {file_path}")
        except Exception as e:
            print(f"[ERROR] Saving failed: {e}")

    return all_questions


def respond_question(
    questions: List[Dict],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    reports: Dict[str, str] = None,
    output_path: str = None,
    report_type: Optional[str] = None
) -> List[Dict]:
    """
    Evaluate MCQs by asking an LLM to answer them, optionally using the associated report text.

    Args:
        questions: List of dicts with keys: question, choices, correct, report_id.
        llm: LLM instance with .run(prompt, json_output=True)
        reports: Dict mapping report_id -> report_text
        with_report: Whether to pass the report to the prompt
        output_path: Optional path to save the output
        source: The source label to use in the output metadata

    Returns:
        List of dicts with answer, confidence, question, correct answer, etc.
    """
    llm = get_llm(provider, model, api_key=load_openai_key())
    start_time = time.time()

    print(f"[INFO] Starting to answer questions ...\n")
    results = []
    # 1. Count how many questions are associated with each report
    report_question_counts = Counter(q["report_id"] for q in questions)
    counts = list(report_question_counts.values())
    report_type = report_type or "wo"

    # 2. Determine the mode of question counts per report
    try:
        questions_mode = mode(counts)
    except StatisticsError:
        # Fallback when no unique mode; pick most common
        questions_mode = Counter(counts).most_common(1)[0][0]

    for q in tqdm(questions, desc= "Answering Questions"):
        if report_type in ("gt", "llm"):
            report_text = reports.get(q["report_id"], "")
            print(f"[INFO] Answering questions with access to report text ...\n")

        else:
            report_text = None
            print(f"[INFO] Answering questions without access to report text ...\n")
            
        prompt = answer_prompt_template(q["question"], q["choices"], report=report_text)
        print(prompt)

        try:
            response = llm.run(prompt, json_output=True)
            results.append({
                "report_id": q["report_id"],
                "question": q["question"],
                "question_source": q["question_source"],
                "choices": q["choices"],
                "answer": q["answer"],
                "prediction": response.get("answer", None),
                "confidence": response.get("confidence", None),
                "report_source": report_type
            })
        except Exception as e:
            print(f"[Error] report {q['report_id']} - {e}")
            results.append({
                "report_id": q["report_id"],
                "question": q["question"],
                "question_source": q["question_source"],
                "choices": q["choices"],
                "answer": q["answer"],
                "prediction": None,
                "confidence": None,
                "report_source": report_type,
                "error": str(e)
            })
    total_time = time.time() - start_time
    print(f"\n[INFO] Question generation completed in {total_time:.2f} seconds.")
    print(f"[INFO] Total questions answers: {len(results)}")
    if output_path:
        # Ensure output_path is a directory
        os.makedirs(output_path, exist_ok=True)
        filename = f"ans{questions_mode}_rs{report_type}_{provider}_{model.replace('/', '-')}.json"
        file_path = os.path.join(output_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved to {file_path}")

    return results
