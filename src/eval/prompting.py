import time
import os
import json
import re
import csv
import pandas as pd
import glob

from typing import List, Callable, Optional, Union, Dict, Any, Tuple
from tqdm import tqdm
from collections import Counter
from statistics import mode, StatisticsError


def load_openai_key():
    """
    Load OpenAI API key from a Beaker configuration file.

    Args:
        beaker_conf_path (str): Path to the .beaker.conf file.

    Returns:
        str: OpenAI API key.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    # print("Loaded API key (first 10 chars):", openai_api_key[:10] + "...")

    return openai_api_key


def load_reports(
    csv_path: str,
    report_column: str = "Ground Truth",
    task_id: str = "ID",
    prompt: Optional[str] = "Prompt",
) -> List[Tuple[str, str, Optional[str], str]]:
    """
    Load structured reports from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the reports.
    report_column : str, optional
        Column name containing the main report text. Default is "Ground Truth".
    task_id : str, optional
        Column name containing the unique ID for each task. Default is "ID".
    prompt : str, optional
        Column name containing associated prompt/context. Default is "Prompt".
        If the column does not exist, None is returned for all rows.

    Returns
    -------
    List[Tuple[str, str, Optional[str], str]]
        A list of tuples for each valid report:
        (
            task_idx : str              - Task ID (periods removed),
            report_text : str           - Report text,
            report_prompt : str or None - Prompt text if exists,
            source : str                - Report source. Lowercased, space-stripped report column name.
        )
    """
    reports = []
    source = report_column

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize headers for safe lookup
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for i, row in enumerate(reader, start=1):
            # Safe retrieval of columns
            raw_task_id = row.get(task_id)
            report_text = row.get(report_column)
            report_prompt = row.get(prompt) if prompt in row else None

            if not raw_task_id:
                print(f"Warning: Missing {task_id} in row {i}. Skipping row.")
                continue

            if not report_text or not report_text.strip():
                continue

            #task_idx = raw_task_id.replace(".", "")
            source = source.lower().replace(" ", "")
            #reports.append((task_idx, report_text, report_prompt, source))
            reports.append((raw_task_id, report_text, report_prompt, source))

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
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )

        content = resp.choices[0].message.content.strip()
        if json_output:
            try:
                return json.loads(content)
            except Exception:
                content_clean = re.sub(
                    r"^```json\s*|\s*```$", "", content, flags=re.DOTALL
                ).strip()
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


def load_benchmark_breakdown(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark breakdown CSV with merged Task ID and Context columns.

    Parameters
    ----------
    csv_path : str
        Path to the benchmark breakdown CSV file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing:
        - "task_id": str - Task ID
        - "context": str - Context text
        - "keypoint_num": int - Keypoint number
        - "ground_truth_keypoint": str - Ground truth keypoint text
        - "fine_grained_prompt": str - Fine-grained prompt/question
    """
    tasks = []
    current_task_id = None
    current_context = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize headers
        reader.fieldnames = [h.strip() if h else h for h in reader.fieldnames]

        for row in reader:
            # Handle merged cells: if Task ID is present, update current task
            task_id = row.get("Task ID", "").strip()
            if task_id:
                current_task_id = task_id

            # Handle merged cells: if Context is present, update current context
            context = row.get("Context", "").strip()
            if context:
                current_context = context

            # Skip rows without required fields
            if not current_task_id or not current_context:
                continue

            keypoint_num = row.get("Keypoints", "").strip()
            ground_truth = row.get("Ground Truth Keypoint", "").strip()
            prompt = row.get("Fine-grained Prompt", "").strip()

            # Skip rows without keypoint data
            if not keypoint_num or not ground_truth:
                continue

            try:
                keypoint_num = int(keypoint_num)
            except ValueError:
                continue

            tasks.append(
                {
                    "task_id": current_task_id,
                    "context": current_context,
                    "keypoint_num": keypoint_num,
                    "ground_truth_keypoint": ground_truth,
                    "fine_grained_prompt": prompt,
                }
            )

    return tasks


def batched_keypoints_template(
    task_id: str, context: str, keypoints: List[Dict[str, Any]]
) -> str:
    """
    Template for generating MCQs from batched keypoints.

    Parameters
    ----------
    task_id : str
        Task ID for this batch.
    context : str
        The context/background information for the task.
    keypoints : List[Dict[str, Any]]
        List of keypoint dictionaries, each containing:
        - "keypoint_num": int
        - "keypoint": str

    Returns
    -------
    str
        Formatted prompt for LLM.
    """
    # Format keypoints for display
    keypoints_text = "\n".join(
        [f"Keypoint {kp['keypoint_num']}: {kp['keypoint']}" for kp in keypoints]
    )

    return f"""
You are generating evaluation questions directly and ONLY from the keypoints listed below.

DO NOT use Discovera reports, Biomni reports, or any other text.

The correct answer MUST be deducible solely from the content of the keypoint.

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

--- BEGIN KEYPOINTS ---
{keypoints_text}
--- END KEYPOINTS ---

For each keypoint:

1. The question must test the truth of the keypoint exactly as written.
   a. Do NOT alter, reinterpret, invert, generalize, or narrow the meaning.
   b. Do NOT ask questions that the answering model could answer using its own
      report instead of the keypoint.

2. Avoid any question format that could be answered by reasoning from general
   biology knowledge alone. The model must be forced to rely on the keypoint
   itself.

3. The question must not require external domain knowledge (e.g., what a gene
   normally does). Everything the model needs must be explicitly stated in the
   keypoint.

4. Forbidden question types:
   - "Which of the following is NOT…"
   - "All of the following EXCEPT…"
   - Questions about numerical values, quantities, dates, sample sizes.
   - Questions requiring knowledge beyond the keypoint text.

5. Allowed style:
   - Direct restatement (best for accuracy)
   - Simple conceptual paraphrases that retain identical meaning
   - "According to the description…" → forces use of the keypoint
   - "In this analysis…" → avoids contamination from the model's own report

6. Each question must have EXACTLY four answer choices, formatted as:
   A. <choice text>
   B. …
   C. …
   D. …

7. Correct answers MUST be randomly distributed among A–D with NO patterns:
   - Avoid consecutive identical answers (e.g., don't put 3+ same answers in a row).
   - Ensure roughly equal distribution across all four choices (A, B, C, D).

8. You may include "None of the above" sparingly (max 10% of questions) —
   and it may be correct or incorrect.

9. Output format (strict):
   A JSON list of objects:
   [
     {{
       "task_id": "{task_id}",
       "question_id": "<keypoint_num>",
       "question_source": "groundtruth",
       "question": "<string>",
       "choices": [
         "A. <string>",
         "B. <string>",
         "C. <string>",
         "D. <string>"
       ],
       "answer": "<A/B/C/D>"
     }}
   ]

Generate exactly {len(keypoints)} questions, one for each keypoint listed above.
""".strip()


def multiple_questions_template(report_text: str, prompt: str, n: int) -> str:
    return f"""
        You are a biomedical researcher. Your task is to generate UP TO {n}
        multiple-choice questions strictly based on the ANALYSIS provided
        below.
        The ANALYSIS answers the following guiding question: {prompt}.
        --- BEGIN ANALYSIS ---
        {report_text}
        --- END ANALYSIS ---

        Instructions:
        1. First, identify the most important findings, mechanisms, or
           insights in the analysis that directly contribute to answering
           the main question above.
        2. Formulate questions only from these central points. Exclude background details or minor observations.
        3. Avoid questions that rely on numerical values, dates, or excessively specific facts.
        4. Do NOT use knowledge outside of the analysis.
        5. Ensure all questions are conceptual, inferential, or analytical,
           emphasizing reasoning, implications, and relationships.
        6. Each question must have exactly four answer choices, formatted as follows:
            A. <choice text>
            B. <choice text>
            C. <choice text>
            D. <choice text>
        7. The correct answer should be randomly distributed among A, B, C, or D — avoid always placing it as "A".
        8. Occasionally, you may include "None of the above" as one of the
           four choices if appropriate — but do not overuse it. It can be
           either correct or incorrect.
        9. If fewer than {n} high-quality questions can be generated, provide only those that meet the criteria.
        10. Ensure the output is valid JSON.
                    
        Output format:
        A JSON list of up to {n} objects, where each object contains:
        - "question": string
        - "choices": list of 4 strings (each starting with "A. ", "B. ", etc.)
        - "correct": string (one of "A", "B", "C", "D")

        Example output:
        [
        {{
            "question": "What pathway was most enriched in the analysis?",
            "choices": [
            "A. Pathway A",
            "B. Pathway B",
            "C. Pathway C",
            "D. Pathway D"
            ],
            "correct": "B"
        }}
        ]
        """.strip()


def answer_prompt_template(question: str, choices: List[str], report: str) -> str:
    if report:
        prompt = f"""
        You are a biomedical domain expert. Use ONLY the information provided in the ANALYSIS below to answer the question.
        You are strictly prohibited from using world knowledge, prior training knowledge, or domain expertise.
        You must behave as if this report is the only biological text you have ever seen.
        Do not make assumptions.
        
        --- BEGIN ANALYSIS ---
        {report}
        --- END ANALYSIS ---

        QUESTION:
        {question}

        CHOICES:
        {choices[0]}
        {choices[1]}
        {choices[2]}
        {choices[3]}


        INSTRUCTIONS:
        - Choose the correct answer ONLY if it is explicitly supported or can be directly inferred from the ANALYSIS.
        - Do NOT use outside knowledge, prior experience, or speculation.
        - **For each choice A–D:**
            - **Check if the report supports it ("Supported"), contradicts it ("Contradicted"), or does not mention it ("Not mentioned").**
            - **Then pick the choice with the strongest textual support.**
            - **If none are supported, answer "E" (meaning I don't know).**

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
        {choices[0]}
        {choices[1]}
        {choices[2]}
        {choices[3]}

        Instructions:
        - Answer the question based on your biomedical knowledge.
        - Provide the best choice letter ("A", "B", "C", or "D") and a confidence score between 0 and 1.

        Respond ONLY in this JSON format:
        {{
          "answer": "<A/B/C/D>",
          "confidence": <float between 0 and 1>
        }}
        """.strip()

    return prompt


def generate_questions(
    reports: List[tuple],
    prompt_template: Callable[[str, int], str],
    provider: str = "openai",
    model: str = "gpt-4o",
    num_questions: Union[int, Dict[str, int]] = 20,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice questions (MCQs) from a collection of reports.

    This function takes in one or more text-based reports and a prompt template,
    then uses a specified LLM provider/model to generate MCQs. The number of
    questions can be fixed across all reports or customized per report.

    Args:
        reports (List[tuple]):
            A list of tuples where each tuple represents a report. Expected format:
            (report_id: str, report_text: str).

        prompt_template (Callable[[str, int], str]):
            A callable that formats the input text and number of questions into
            a prompt string for the LLM. Signature: (report_text, num_questions) → prompt.

        provider (str, optional):
            The LLM service provider to use (e.g., "openai", "anthropic").
            Defaults to "openai".

        model (str, optional):
            The model name to use for question generation.
            Defaults to "gpt-4o".

        num_questions (Union[int, Dict[str, int]], optional):
            - If int: generate the same number of questions for each report.
            - If dict: keys are report_ids, values are custom question counts.
            Defaults to 20.

        output_path (Optional[str], optional):
            If provided, the generated questions will also be written to this
            path as a JSON file. Defaults to None.


    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries, each containing:
            - "report_id" (str): ID of the source report.
            - "question" (str): The generated question.
            - "options" (List[str]): Answer choices for the question.
            - "answer" (str): The correct answer.
            - "question_source" (str, optional): The provided `source` value. Defining that the
            questions where based on either GT or LLM-generated reports.

    Raises:
        ValueError: If num_questions is a dict but a report_id is missing.
        RuntimeError: If the LLM call fails or response parsing fails.
    """

    llm = get_llm(provider, model, api_key=load_openai_key())
    start_time = time.time()

    print(f"[INFO] Generating questions for {len(reports)} reports...")
    all_questions = []

    # Determine if we're using custom counts
    custom_counts = isinstance(num_questions, dict)

    for report in tqdm(reports, desc="Processing Reports"):
        task_id = report[0]
        report_text = report[1]
        prompt = report[2]
        type_report = report[3]

        # Decide per-report question count
        if custom_counts:
            count = num_questions.get(task_id)
            if count is None:
                print(f"[WARNING] No num_questions specified for {task_id}. Skipping.")
                continue
        else:
            count = num_questions  # single fixed integer

        prompt = prompt_template(report_text, prompt, count)

        try:
            mcqs = llm.run(prompt, json_output=True)
            if isinstance(mcqs, list):
                for q in mcqs:
                    all_questions.append(
                        {
                            "task_id": task_id,
                            "question": q.get("question", ""),
                            "choices": q.get("choices", []),
                            "answer": q.get("correct", ""),
                            "question_source": type_report,
                        }
                    )
            else:
                print(f"[WARNING] Unexpected format for {task_id}")
        except Exception as e:
            print(f"[ERROR] Report {task_id} failed: {e}")

    elapsed = time.time() - start_time
    print(
        f"[INFO] Completed in {elapsed:.2f} seconds. Total questions: {len(all_questions)}"
    )

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        suffix = str(num_questions) if not custom_counts else "cus"
        filename = (
            f"qs{suffix}_rs{type_report}_{provider}_{model.replace('/', '-')}.json"
        )
        file_path = os.path.join(output_path, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_questions, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved to {file_path}")
        except Exception as e:
            print(f"[ERROR] Saving failed: {e}")

    return all_questions


def generate_questions_from_breakdown(
    csv_path: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    output_path: Optional[str] = None,
    task_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice questions from benchmark breakdown CSV.
    Generates one MCQ per keypoint, batched by task (context + list of keypoints).

    This function loads the benchmark breakdown CSV, groups keypoints by task_id,
    and generates questions in batches using the Context and all keypoints for
    each task.

    Args:
        csv_path (str):
            Path to the benchmark breakdown CSV file.

        provider (str, optional):
            The LLM service provider to use (e.g., "openai", "anthropic").
            Defaults to "openai".

        model (str, optional):
            The model name to use for question generation.
            Defaults to "gpt-4o".

        output_path (Optional[str], optional):
            If provided, the generated questions will also be written to this
            path as a JSON file. Defaults to None.

        task_filter (Optional[List[str]], optional):
            If provided, only generate questions for tasks with IDs in this list.
            Defaults to None (process all tasks).

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries, each containing:
            - "task_id" (str): Task ID from the CSV
            - "question_id" (str): Keypoint number as string
            - "question_source" (str): "groundtruth"
            - "question" (str): The generated question
            - "choices" (List[str]): Answer choices for the question
            - "answer" (str): The correct answer (A, B, C, or D)
            - "ground_truth_keypoint" (str): The ground truth keypoint text

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        RuntimeError: If the LLM call fails or response parsing fails.
    """
    # Load benchmark breakdown data
    tasks = load_benchmark_breakdown(csv_path)

    if not tasks:
        print(f"[WARNING] No valid tasks found in {csv_path}")
        return []

    # Filter tasks if task_filter is provided
    if task_filter:
        tasks = [t for t in tasks if t["task_id"] in task_filter]
        print(f"[INFO] Filtered to {len(tasks)} keypoints matching filter")

    # Group keypoints by task_id
    task_groups = {}
    for task in tasks:
        task_id = task["task_id"]
        if task_id not in task_groups:
            task_groups[task_id] = {"context": task["context"], "keypoints": []}
        task_groups[task_id]["keypoints"].append(
            {
                "keypoint_num": task["keypoint_num"],
                "keypoint": task["ground_truth_keypoint"],
            }
        )

    # Sort keypoints by keypoint_num within each task
    for task_id in task_groups:
        task_groups[task_id]["keypoints"].sort(key=lambda x: x["keypoint_num"])

    llm = get_llm(provider, model, api_key=load_openai_key())
    start_time = time.time()

    num_tasks = len(task_groups)
    total_keypoints = sum(len(g["keypoints"]) for g in task_groups.values())
    print(
        f"[INFO] Generating questions for {total_keypoints} keypoints "
        f"from {num_tasks} tasks (batched by task)..."
    )
    all_questions = []

    # Process each task as a batch
    for task_id, task_data in tqdm(task_groups.items(), desc="Processing Tasks"):
        context = task_data["context"]
        keypoints = task_data["keypoints"]

        # Generate prompt using the batched template
        llm_prompt = batched_keypoints_template(task_id, context, keypoints)

        try:
            response = llm.run(llm_prompt, json_output=True)

            # Response should be a list of questions
            if not isinstance(response, list):
                print(
                    f"[WARNING] Expected list for task {task_id}, got {type(response)}"
                )
                continue

            # Create a mapping from question_id to keypoint data
            keypoint_map = {str(kp["keypoint_num"]): kp for kp in keypoints}

            # Process each question in the response
            task_questions = []
            for q in response:
                question_id = str(q.get("question_id", ""))
                if question_id not in keypoint_map:
                    print(
                        f"[WARNING] Question ID {question_id} not found in "
                        f"keypoints for task {task_id}"
                    )
                    continue

                kp_data = keypoint_map[question_id]
                question_dict = {
                    "task_id": q.get("task_id", task_id),
                    "question_id": question_id,
                    "question_source": q.get("question_source", "groundtruth"),
                    "question": q.get("question", ""),
                    "choices": q.get("choices", []),
                    "answer": q.get("answer", ""),
                    "ground_truth_keypoint": kp_data.get("keypoint", ""),
                }
                task_questions.append(question_dict)
                all_questions.append(question_dict)

            # Verify we got the expected number of questions
            if len(response) != len(keypoints):
                print(
                    f"[WARNING] Task {task_id}: Expected {len(keypoints)} "
                    f"questions, got {len(response)}"
                )

            # Save questions for this task
            if output_path and task_questions:
                task_output_dir = os.path.join(output_path, task_id)
                os.makedirs(task_output_dir, exist_ok=True)

                # Format model name: replace slashes with hyphens, hyphens with underscores, keep dots
                model_name = model.replace("/", "-").replace("-", "_")
                num_questions = len(task_questions)
                filename = (
                    f"qs{num_questions}_rsgroundtruth_{provider}_{model_name}.json"
                )
                file_path = os.path.join(task_output_dir, filename)

                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(task_questions, f, indent=2, ensure_ascii=False)
                    print(
                        f"[INFO] Saved {num_questions} questions for task "
                        f"{task_id} to {file_path}"
                    )
                except Exception as e:
                    print(f"[ERROR] Saving failed for task {task_id}: {e}")

        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            continue

    elapsed = time.time() - start_time
    print(
        f"[INFO] Completed in {elapsed:.2f} seconds. "
        f"Total questions: {len(all_questions)}"
    )

    return all_questions


def respond_questions_from_breakdown(
    task_ids: List[str],
    csv_path: str,
    questions_dir: str,
    answers_dir: str,
    report_columns: List[str],
    provider: str = "openai",
    model: str = "gpt-4o",
) -> None:
    """
    Automatically traverse task IDs, load questions, and answer them using
    reports from CSV.

    Args:
        task_ids: List of task IDs to process (e.g., ["1", "2.1", "3.1"])
        csv_path: Path to benchmark_verified.csv
        questions_dir: Directory containing question JSON files
                      (e.g., "output/questions_new")
        answers_dir: Directory to save answered questions
                    (e.g., "output/answers_new")
        report_columns: List of CSV column names to use as reports
                       (e.g., ["Discovera (o4-mini)", "LLM (o4-mini)", "Biomni (o4-mini)"])
        provider: LLM provider (e.g., "openai")
        model: LLM model name (e.g., "gpt-4o")
    """
    llm = get_llm(provider, model, api_key=load_openai_key())

    # Load CSV data
    print(f"[INFO] Loading reports from {csv_path}...")
    csv_data = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            task_id = row.get("ID", "").strip()
            if task_id:
                csv_data[task_id] = row

    print(f"[INFO] Found {len(csv_data)} tasks in CSV")

    # Process each task ID
    for task_id in tqdm(task_ids, desc="Processing Tasks"):
        # Find question files for this task
        task_questions_dir = os.path.join(questions_dir, task_id)
        if not os.path.exists(task_questions_dir):
            print(f"[WARNING] Questions directory not found: {task_questions_dir}")
            continue

        # Find all JSON files in the task directory
        question_files = [
            f for f in os.listdir(task_questions_dir) if f.endswith(".json")
        ]

        if not question_files:
            print(f"[WARNING] No question files found in {task_questions_dir}")
            continue

        # Process each question file
        for question_file in question_files:
            question_path = os.path.join(task_questions_dir, question_file)

            # Load questions
            try:
                with open(question_path, "r", encoding="utf-8") as f:
                    questions = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {question_path}: {e}")
                continue

            if not isinstance(questions, list) or not questions:
                print(f"[WARNING] Invalid questions format in {question_path}")
                continue

            # Get CSV row for this task
            csv_row = csv_data.get(task_id)
            if not csv_row:
                print(f"[WARNING] Task {task_id} not found in CSV")
                continue

            # Answer questions using each report column
            for report_column in report_columns:
                report_text = csv_row.get(report_column, "").strip()
                if not report_text:
                    print(
                        f"[WARNING] Empty report for task {task_id}, "
                        f"column {report_column}"
                    )
                    continue

                # Generate source name from column name
                # e.g., "Discovera (o4-mini)" -> "discovera(o4-mini)"
                source_name = report_column.lower().replace(" ", "")

                # Answer questions
                results = []
                print(
                    f"[INFO] Answering {len(questions)} questions for task "
                    f"{task_id} using {report_column}..."
                )

                for q in tqdm(questions, desc=f"Task {task_id}"):
                    prompt = answer_prompt_template(
                        q["question"], q["choices"], report=report_text
                    )
                    try:
                        response = llm.run(prompt, json_output=True)
                        results.append(
                            {
                                "task_id": q.get("task_id", task_id),
                                "question_id": q.get("question_id", ""),
                                "question": q.get("question", ""),
                                "question_source": q.get(
                                    "question_source", "groundtruth"
                                ),
                                "choices": q.get("choices", []),
                                "answer": q.get("answer", ""),
                                "prediction": response.get("answer", None),
                                "confidence": response.get("confidence", None),
                                "report_source": source_name,
                            }
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Failed to answer question for task "
                            f"{task_id}: {e}"
                        )
                        results.append(
                            {
                                "task_id": q.get("task_id", task_id),
                                "question_id": q.get("question_id", ""),
                                "question": q.get("question", ""),
                                "question_source": q.get(
                                    "question_source", "groundtruth"
                                ),
                                "choices": q.get("choices", []),
                                "answer": q.get("answer", ""),
                                "prediction": None,
                                "confidence": None,
                                "report_source": source_name,
                                "error": str(e),
                            }
                        )

                # Save results
                if results:
                    task_answers_dir = os.path.join(answers_dir, task_id)
                    os.makedirs(task_answers_dir, exist_ok=True)

                    # Extract number of questions from question filename
                    # e.g., "qs12_rsgroundtruth_openai_gpt_5_nano.json" -> "qs12"
                    qs_match = question_file.split("_")[0]  # "qs12"
                    num_questions = (
                        qs_match.replace("qs", "")
                        if qs_match.startswith("qs")
                        else str(len(results))
                    )

                    # Format model name
                    model_name = model.replace("/", "-").replace("-", "_")

                    # Create answer filename
                    answer_filename = f"ans{num_questions}_rs{source_name}_{provider}_{model_name}.json"
                    answer_path = os.path.join(task_answers_dir, answer_filename)

                    try:
                        with open(answer_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        print(f"[INFO] Saved {len(results)} answers to {answer_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to save answers to {answer_path}: {e}")

    print("[INFO] Completed processing all tasks")


def respond_question(
    questions: List[Dict],
    provider: str = "openai",
    model: str = "gpt-4o",
    reports: list[tuple[str, str]] = None,
    output_path: str = None,
) -> List[Dict]:
    """
    Evaluate MCQs by asking an LLM to answer them, optionally using the associated report text.

    Args:
        questions: List of dicts with keys: question, choices, correct, report_id.
        provider: LLM provider (e.g., "openai").
        model: LLM model name (e.g., "gpt-4o").
        reports: List of tuples (report_key, report_text, prompt, source).
        output_path: Optional path to save the output
    Returns:
        List of dicts with answer, confidence, question, correct answer, etc.
    """
    llm = get_llm(provider, model, api_key=load_openai_key())

    layered_reports = {r[3]: dict() for r in reports}
    for report in reports:
        layered_reports[report[3]][report[0]] = report[1]

    print("[INFO] Starting to answer questions ...\n")
    # 1. Count how many questions are associated with each report
    report_question_counts = Counter(q["task_id"] for q in questions)
    counts = list(report_question_counts.values())
    # 2. Determine the mode of question counts per report
    try:
        questions_mode = mode(counts)
    except StatisticsError:
        # Fallback when no unique mode; pick most common
        questions_mode = Counter(counts).most_common(1)[0][0]
    for source, report_dict in layered_reports.items():
        results = []
        print(f"[INFO] Answering questions for report source: {source} ...\n")
        for q in tqdm(questions, desc="Answering Questions"):
            report = report_dict.get(str(q["task_id"]), "")
            prompt = answer_prompt_template(q["question"], q["choices"], report=report)
            try:
                response = llm.run(prompt, json_output=True)
                results.append(
                    {
                        "task_id": q["task_id"],
                        "question": q["question"],
                        "question_source": q["question_source"],
                        "choices": q["choices"],
                        "answer": q["answer"],
                        "prediction": response.get("answer", None),
                        "confidence": response.get("confidence", None),
                        "report_source": source,
                    }
                )
            except Exception as e:
                print(f"[Error] report {q['task_id']} - {e}")
                results.append(
                    {
                        "task_id": q["task_id"],
                        "question": q["question"],
                        "question_source": q["question_source"],
                        "choices": q["choices"],
                        "answer": q["answer"],
                        "prediction": None,
                        "confidence": None,
                        "report_source": source,
                        "error": str(e),
                    }
                )

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            filename = f"ans{questions_mode}_rs{source}_{provider}_{model.replace('/', '-')}.json"
            file_path = os.path.join(output_path, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Saved to {file_path}")


def respond_questions_with_finegrained_reports(
    questions_dir: str,
    reports_dir: str,
    answers_dir: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    task_ids: Optional[List[str]] = None,
) -> None:
    """
    Answer questions using fine-grained reports stored as individual JSON files.

    This function reads questions from the questions directory, matches them with
    fine-grained report files (task_{task_id}_{keypoint_id}.json), and answers
    each question using the corresponding report's output_text.

    Args:
        questions_dir (str):
            Directory containing question JSON files, organized by task_id.
            Expected structure: questions_dir/{task_id}/*.json

        reports_dir (str):
            Directory containing fine-grained report JSON files.
            Expected filename format: task_{task_id}_{keypoint_id}.json
            Each file should contain an "output_text" field with the report.

        answers_dir (str):
            Directory to save answered questions.
            Output structure: answers_dir/{task_id}/ans{num}_rsfinegrained_{provider}_{model}.json

        provider (str, optional):
            The LLM service provider to use (e.g., "openai", "anthropic").
            Defaults to "openai".

        model (str, optional):
            The model name to use for answering questions.
            Defaults to "gpt-4o".

        task_ids (Optional[List[str]], optional):
            If provided, only process questions for tasks with IDs in this list.
            Defaults to None (process all tasks found in questions_dir).

    Returns:
        None: Results are saved to files in answers_dir.

    Example:
        >>> respond_questions_with_finegrained_reports(
        ...     questions_dir="output/questions_new",
        ...     reports_dir="outputs/Discovera",
        ...     answers_dir="output/answers_finegrained",
        ...     provider="openai",
        ...     model="gpt-5-nano"
        ... )
    """
    llm = get_llm(provider, model, api_key=load_openai_key())

    # Determine which tasks to process
    if task_ids is None:
        task_dirs = [
            d
            for d in os.listdir(questions_dir)
            if os.path.isdir(os.path.join(questions_dir, d))
        ]
    else:
        task_dirs = task_ids

    print(f"[INFO] Processing {len(task_dirs)} tasks from {questions_dir}...")

    for task_id in tqdm(task_dirs, desc="Processing Tasks"):
        task_questions_dir = os.path.join(questions_dir, task_id)
        if not os.path.isdir(task_questions_dir):
            print(f"[WARNING] Questions directory not found: {task_questions_dir}")
            continue

        # Find all JSON question files in this task directory
        question_files = [
            f for f in os.listdir(task_questions_dir) if f.endswith(".json")
        ]

        if not question_files:
            print(f"[WARNING] No question files found in {task_questions_dir}")
            continue

        # Process each question file
        for question_file in question_files:
            question_path = os.path.join(task_questions_dir, question_file)

            # Load questions
            try:
                with open(question_path, "r", encoding="utf-8") as f:
                    questions = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load {question_path}: {e}")
                continue

            if not isinstance(questions, list) or not questions:
                print(f"[WARNING] Invalid question format in {question_path}")
                continue

            # Answer questions using fine-grained reports
            results = []
            print(
                f"[INFO] Answering {len(questions)} questions for task {task_id} "
                f"using fine-grained reports..."
            )

            for q in tqdm(questions, desc=f"Task {task_id}"):
                question_id = str(q.get("question_id", ""))
                if not question_id:
                    print(
                        f"[WARNING] Missing question_id in question from {question_path}"
                    )
                    continue

                # Construct fine-grained report filename
                # Format: task_{task_id}_{keypoint_id}.json
                report_filename = f"task_{task_id}_{question_id}.json"
                report_path = os.path.join(reports_dir, report_filename)

                # Load fine-grained report
                report_text = ""
                if os.path.exists(report_path):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            report_data = json.load(f)
                        report_text = report_data.get("output_text", "").strip()
                    except Exception as e:
                        print(
                            f"[WARNING] Failed to load report {report_path}: {e}"
                        )
                else:
                    print(
                        f"[WARNING] Fine-grained report not found: {report_path}"
                    )

                # Answer question using the report
                prompt = answer_prompt_template(
                    q.get("question", ""), q.get("choices", []), report=report_text
                )

                try:
                    response = llm.run(prompt, json_output=True)
                    results.append(
                        {
                            "task_id": q.get("task_id", task_id),
                            "question_id": question_id,
                            "question": q.get("question", ""),
                            "question_source": q.get(
                                "question_source", "groundtruth"
                            ),
                            "choices": q.get("choices", []),
                            "answer": q.get("answer", ""),
                            "prediction": response.get("answer", None),
                            "confidence": response.get("confidence", None),
                            "report_source": "finegrained",
                            "report_file": report_filename,
                        }
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Failed to answer question for task {task_id}, "
                        f"question_id {question_id}: {e}"
                    )
                    results.append(
                        {
                            "task_id": q.get("task_id", task_id),
                            "question_id": question_id,
                            "question": q.get("question", ""),
                            "question_source": q.get(
                                "question_source", "groundtruth"
                            ),
                            "choices": q.get("choices", []),
                            "answer": q.get("answer", ""),
                            "prediction": None,
                            "confidence": None,
                            "report_source": "finegrained",
                            "report_file": report_filename,
                            "error": str(e),
                        }
                    )

            # Save results
            if results:
                task_answers_dir = os.path.join(answers_dir, task_id)
                os.makedirs(task_answers_dir, exist_ok=True)

                # Extract number of questions from question filename
                # e.g., "qs12_rsgroundtruth_openai_gpt_5_nano.json" -> "qs12"
                qs_match = question_file.split("_")[0]  # "qs12"
                num_questions = (
                    qs_match.replace("qs", "")
                    if qs_match.startswith("qs")
                    else str(len(results))
                )

                # Format model name
                model_name = model.replace("/", "-").replace("-", "_")

                # Create answer filename
                answer_filename = (
                    f"ans{num_questions}_rsfinegrained_{provider}_{model_name}.json"
                )
                answer_path = os.path.join(task_answers_dir, answer_filename)

                try:
                    with open(answer_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Saved {len(results)} answers to {answer_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save answers to {answer_path}: {e}")

    print("[INFO] Completed processing fine-grained reports.")


def generate_score_comparison_table(
    answers_dir: str, task_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read answer files from the answers directory and generate a score
    comparison table showing accuracy breakdowns by task, method, and model.

    Args:
        answers_dir: Directory containing answer JSON files
                    (e.g., "output/answers_new")
        task_ids: Optional list of task IDs to process. If None, processes
                 all tasks found in the directory.

    Returns:
        pd.DataFrame: Comparison table with columns:
            - task_id: Task ID
            - report_source: Report source (e.g., "discovera(o4-mini)")
            - provider: LLM provider (e.g., "openai")
            - model: Model name (e.g., "gpt_5_nano")
            - total_questions: Total number of questions
            - correct: Number of correct answers
            - accuracy: Accuracy score (0-1)
            - avg_confidence: Average confidencse score
            - weighted_score: Weighted score calculated as (accuracy * total_questions) /
              sum(total_questions) for each report_source/provider/model group
    """
    all_results = []

    # Get all task directories
    if task_ids is None:
        # Find all task directories
        task_dirs = [
            d
            for d in os.listdir(answers_dir)
            if os.path.isdir(os.path.join(answers_dir, d))
        ]
    else:
        task_dirs = task_ids

    print(f"[INFO] Processing {len(task_dirs)} tasks from {answers_dir}...")

    for task_id in tqdm(task_dirs, desc="Processing Tasks"):
        task_dir = os.path.join(answers_dir, task_id)
        if not os.path.isdir(task_dir):
            continue

        # Find all JSON answer files in this task directory
        answer_files = glob.glob(os.path.join(task_dir, "*.json"))

        for answer_file in answer_files:
            try:
                with open(answer_file, "r", encoding="utf-8") as f:
                    answers = json.load(f)

                if not isinstance(answers, list) or not answers:
                    continue

                # Extract metadata from filename
                # Format: ans{num}_rs{source}_{provider}_{model}.json
                filename = os.path.basename(answer_file)
                parts = filename.replace(".json", "").split("_")

                # Parse filename components
                report_source = None
                provider = None
                model = None

                # Find report_source (starts with "rs")
                for i, part in enumerate(parts):
                    if part.startswith("rs"):
                        report_source = part[2:]  # Remove "rs" prefix
                        if i + 1 < len(parts):
                            provider = parts[i + 1]
                        if i + 2 < len(parts):
                            model = "_".join(parts[i + 2:])  # Join remaining parts
                        break

                if not report_source or not provider or not model:
                    # Try alternative parsing
                    if len(parts) >= 4:
                        report_source = (
                            parts[1].replace("rs", "")
                            if parts[1].startswith("rs")
                            else parts[1]
                        )
                        provider = parts[2] if len(parts) > 2 else None
                        model = "_".join(parts[3:]) if len(parts) > 3 else None

                # Calculate scores
                total_questions = len(answers)
                correct = 0
                confidences = []

                for ans in answers:
                    if ans.get("answer") and ans.get("prediction"):
                        if str(ans["answer"]).upper() == str(ans["prediction"]).upper():
                            correct += 1
                    if ans.get("confidence") is not None:
                        confidences.append(ans["confidence"])

                accuracy = correct / total_questions if total_questions > 0 else 0.0
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else None
                )

                all_results.append(
                    {
                        "task_id": task_id,
                        "report_source": report_source,
                        "provider": provider,
                        "model": model,
                        "total_questions": total_questions,
                        "correct": correct,
                        "accuracy": accuracy,
                        "avg_confidence": avg_confidence,
                        "filename": filename,
                    }
                )

            except Exception as e:
                print(f"[WARNING] Failed to process {answer_file}: {e}")
                continue

    # Create DataFrame
    if not all_results:
        print("[WARNING] No results found. Returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "task_id",
                "report_source",
                "provider",
                "model",
                "total_questions",
                "correct",
                "accuracy",
                "avg_confidence",
            ]
        )

    df = pd.DataFrame(all_results)

    # Sort by task_id, then report_source, then model
    df = df.sort_values(
        by=["task_id", "report_source", "model"], ascending=[True, True, True]
    ).reset_index(drop=True)

    print(f"[INFO] Generated comparison table with {len(df)} rows")
    return df


def extract_keypoints_prompt(ground_truth_text: str) -> str:
    """
    Generate prompt for extracting keypoints from ground truth text.

    Args:
        ground_truth_text: The ground truth text to extract keypoints from.

    Returns:
        Formatted prompt string.
    """
    return f"""
You are analyzing a biomedical ground truth text. Extract the key scientific claims or findings as distinct keypoints.

--- BEGIN GROUND TRUTH TEXT ---
{ground_truth_text}
--- END GROUND TRUTH TEXT ---

Instructions:
1. Break down the text into distinct, atomic keypoints (claims/findings).
2. Each keypoint should be a single, clear scientific statement.
3. Preserve the exact meaning and scientific accuracy.
4. Number each keypoint sequentially (1, 2, 3, ...).
5. Each keypoint should be self-contained and meaningful on its own.

Output format (JSON):
{{
  "keypoints": [
    "Keypoint 1 text here",
    "Keypoint 2 text here",
    ...
  ]
}}

Generate the keypoints now:
""".strip()


def check_coverage_prompt(keypoint: str, report_text: str, report_name: str) -> str:
    """
    Generate prompt for checking if a keypoint is covered in a report.

    Args:
        keypoint: The keypoint to check.
        report_text: The report text to check against.
        report_name: Name of the report (e.g., "Discovera").

    Returns:
        Formatted prompt string.
    """
    return f"""
You are evaluating whether a specific scientific keypoint is covered in a biomedical report.

--- BEGIN KEYPOINT ---
{keypoint}
--- END KEYPOINT ---

--- BEGIN {report_name} REPORT ---
{report_text}
--- END {report_name} REPORT ---

Task:
1. Determine if the keypoint is covered in the report (YES or NO).
2. If YES, provide specific evidence/quotes from the report that support the keypoint.
3. If NO, explain why it's not covered (missing information, contradicted, etc.).
4. Be precise and cite exact text when possible.

Output format (JSON):
{{
  "covered": "YES" or "NO",
  "evidence": "Specific evidence from the report, including quotes if available. If NO, explain why not covered."
}}

Evaluate now:
""".strip()


def generate_keypoint_coverage_table(
    csv_path: str,
    task_id: str,
    ground_truth_column: str = "Ground Truth",
    report_columns: Optional[List[str]] = None,
    task_id_column: str = "ID",
    provider: str = "openai",
    model: str = "gpt-4o",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a detailed coverage table showing which keypoints from ground truth
    are covered by each report.

    This function:
    1. Extracts keypoints from the ground truth text for a specific task
    2. Checks coverage of each keypoint in each report column
    3. Generates evidence/explanations for coverage decisions
    4. Returns a formatted table

    Args:
        csv_path: Path to the CSV file containing ground truth and reports.
        task_id: Task ID to process (e.g., "1", "2.1").
        ground_truth_column: Column name containing ground truth text.
                           Default is "Ground Truth".
        report_columns: List of column names to check coverage against.
            If None, will attempt to auto-detect report columns.
                      Default is None.
        task_id_column: Column name containing task IDs. Default is "ID".
        provider: LLM provider (e.g., "openai"). Default is "openai".
        model: LLM model name (e.g., "gpt-4o"). Default is "gpt-4o".
        output_path: Optional path to save the results as JSON/CSV.
                    Default is None.

    Returns:
        pd.DataFrame: Coverage table with columns:
            - keypoint_num: Keypoint number (1, 2, 3, ...)
            - keypoint: The keypoint text
            - report_source: Name of the report column
            - covered: "YES" or "NO"
            - evidence: Evidence/explanation for coverage decision
            - task_id: Task ID

    Example:
        >>> df = generate_keypoint_coverage_table(
        ...     csv_path="benchmark.csv",
        ...     task_id="1",
        ...     report_columns=["Discovera (o4-mini)", "Biomni (o4-mini)"]
        ... )
        >>> print(df)
    """
    llm = get_llm(provider, model, api_key=load_openai_key())

    # Load CSV data
    print(f"[INFO] Loading data from {csv_path}...")
    csv_data = {}
    all_columns = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        all_columns = list(reader.fieldnames)

        for row in reader:
            raw_task_id = row.get(task_id_column, "").strip()
            if raw_task_id:
                csv_data[raw_task_id] = row

    # Get task data
    task_row = csv_data.get(task_id)
    if not task_row:
        raise ValueError(f"Task ID '{task_id}' not found in CSV")

    # Get ground truth
    ground_truth_text = task_row.get(ground_truth_column, "").strip()
    if not ground_truth_text:
        raise ValueError(
            f"Ground truth text not found for task {task_id} "
            f"in column '{ground_truth_column}'"
        )

    # Auto-detect report columns if not provided
    if report_columns is None:
        # Exclude common non-report columns
        exclude_cols = {
            task_id_column.lower(),
            ground_truth_column.lower(),
            "prompt",
            "context",
            "id",
        }
        report_columns = [
            col
            for col in all_columns
            if col.lower() not in exclude_cols
            and task_row.get(col, "").strip()  # Has content
        ]
        print(
            f"[INFO] Auto-detected {len(report_columns)} report columns: "
            f"{report_columns}"
        )

    if not report_columns:
        raise ValueError("No report columns found or specified")

    # Step 1: Extract keypoints from ground truth
    print(f"[INFO] Extracting keypoints from ground truth for task {task_id}...")
    extract_prompt = extract_keypoints_prompt(ground_truth_text)

    try:
        keypoints_response = llm.run(extract_prompt, json_output=True)
        if isinstance(keypoints_response, dict):
            keypoints = keypoints_response.get("keypoints", [])
        elif isinstance(keypoints_response, list):
            keypoints = keypoints_response
        else:
            raise ValueError(f"Unexpected keypoints format: {type(keypoints_response)}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract keypoints: {e}")

    if not keypoints:
        raise ValueError("No keypoints extracted from ground truth")

    print(f"[INFO] Extracted {len(keypoints)} keypoints")

    # Step 2: Check coverage for each keypoint in each report
    print(f"[INFO] Checking coverage in {len(report_columns)} reports...")
    coverage_results = []

    for keypoint_num, keypoint in enumerate(keypoints, start=1):
        for report_column in report_columns:
            report_text = task_row.get(report_column, "").strip()
            if not report_text:
                coverage_results.append(
                    {
                        "task_id": task_id,
                        "keypoint_num": keypoint_num,
                        "keypoint": keypoint,
                        "report_source": report_column,
                        "covered": "NO",
                        "evidence": "Report column is empty or missing.",
                    }
                )
                continue

            # Generate source name
            report_name = report_column.split("(")[0].strip()

            # Check coverage
            coverage_prompt = check_coverage_prompt(
                keypoint, report_text, report_name
            )

            try:
                coverage_response = llm.run(coverage_prompt, json_output=True)
                if isinstance(coverage_response, dict):
                    covered = coverage_response.get("covered", "NO").upper()
                    evidence = coverage_response.get("evidence", "")
                else:
                    covered = "NO"
                    evidence = f"Unexpected response format: {type(coverage_response)}"
            except Exception as e:
                print(
                    f"[WARNING] Failed to check coverage for keypoint {keypoint_num} "
                    f"in {report_column}: {e}"
                )
                covered = "NO"
                evidence = f"Error during coverage check: {str(e)}"

            coverage_results.append(
                {
                    "task_id": task_id,
                    "keypoint_num": keypoint_num,
                    "keypoint": keypoint,
                    "report_source": report_column,
                    "covered": covered,
                    "evidence": evidence,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(coverage_results)

    # Sort by keypoint_num, then report_source
    df = df.sort_values(
        by=["keypoint_num", "report_source"], ascending=[True, True]
    ).reset_index(drop=True)

    print(f"[INFO] Generated coverage table with {len(df)} rows")

    # Save if output_path is provided
    if output_path:
        output_dir = (
            os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        )
        os.makedirs(output_dir, exist_ok=True)

        # Determine base name without extension
        base_name = output_path
        if output_path.endswith((".csv", ".json")):
            base_name = output_path.rsplit(".", 1)[0]

        # Save as CSV
        csv_output = f"{base_name}.csv"
        df.to_csv(csv_output, index=False, encoding="utf-8")
        print(f"[INFO] Saved CSV to {csv_output}")

        # Save as JSON
        json_output = f"{base_name}.json"
        df.to_json(json_output, orient="records", indent=2, force_ascii=False)
        print(f"[INFO] Saved JSON to {json_output}")

        return df


def format_coverage_table_markdown(
    df: pd.DataFrame, report_source: Optional[str] = None
) -> str:
    """
    Format coverage table DataFrame as a markdown table.

    Args:
        df: Coverage table DataFrame from generate_keypoint_coverage_table.
        report_source: Optional filter to show only one report source.
                      If None, shows all reports.

    Returns:
        Formatted markdown table string.

    Example:
        >>> df = generate_keypoint_coverage_table(...)
        >>> markdown = format_coverage_table_markdown(df, report_source="Discovera")
        >>> print(markdown)
    """
    # Filter by report_source if specified
    if report_source:
        df = df[df["report_source"] == report_source].copy()

    if df.empty:
        return "No data to display."

    # Create markdown table
    lines = []
    lines.append("| # | Ground Truth Keypoint | Covered? | Why / Evidence |")
    lines.append("| --- | --- | --- | --- |")

    for _, row in df.iterrows():
        keypoint_num = row["keypoint_num"]
        keypoint = row["keypoint"]
        covered = row["covered"]
        evidence = row["evidence"]

        # Truncate long keypoints/evidence for display
        max_keypoint_len = 100
        max_evidence_len = 200

        if len(keypoint) > max_keypoint_len:
            keypoint = keypoint[:max_keypoint_len] + "..."
        if len(evidence) > max_evidence_len:
            evidence = evidence[:max_evidence_len] + "..."

        # Escape pipe characters in markdown
        keypoint = keypoint.replace("|", "\\|")
        evidence = evidence.replace("|", "\\|")

        lines.append(
            f"| **{keypoint_num}** | {keypoint} | **{covered}** | {evidence} |"
        )

    return "\n".join(lines)
