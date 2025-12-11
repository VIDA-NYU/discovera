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

            # task_idx = raw_task_id.replace(".", "")
            source = source.lower().replace(" ", "")
            # reports.append((task_idx, report_text, report_prompt, source))
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


def load_benchmark_breakdown_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark breakdown from JSON format.

    Parameters
    ----------
    json_path : str
        Path to the benchmark breakdown JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each containing:
        - "task_id": str - Task ID
        - "context": str - Context text
        - "keypoint_num": int - Keypoint number (from insight "id")
        - "ground_truth_keypoint": str - Insight text
        - "fine_grained_prompt": str - Fine-grained prompt/question (if available)
    """
    tasks = []

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for task_id, task_data in json_data.items():
        context = task_data.get("context", "").strip()
        insights = task_data.get("insights", [])

        if not context or not insights:
            continue

        for insight in insights:
            insight_id = insight.get("id")
            insight_text = insight.get("insight", "").strip()
            fine_grained_prompt = insight.get("fine-grained prompt", "").strip()

            if not insight_text:
                continue

            try:
                keypoint_num = int(insight_id) if insight_id is not None else None
            except (ValueError, TypeError):
                continue

            if keypoint_num is None:
                continue

            tasks.append(
                {
                    "task_id": task_id,
                    "context": context,
                    "keypoint_num": keypoint_num,
                    "ground_truth_keypoint": insight_text,
                    "fine_grained_prompt": fine_grained_prompt,
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


def breakdown_template(report: str) -> str:
    """
    Template for generating keypoints from a report.

    Parameters
    ----------
    report : str
        The report text to break down into keypoints.

    Returns
    -------
    str
        Formatted prompt for LLM.
    """
    return f"""
You are given a biomedical report. Your task is to generate a set of mechanistic key points that capture fine-grained causal or functional insights from the report.
Instructions:
- Produce no more than 12 mechanistic keypoints.
- Do not include any specific counts, percentages, fold-changes, or sample numbers even if they appear in the report.
- Focus on core mechanistic findings, not general observations.
- Each keypoint must be concise, fine-grained, and biologically meaningful.

--- BEGIN REPORT ---
{report}
--- END REPORT ---

Output format (strict JSON):
[
  {{
    "keypoint_num": 1,
    "keypoint": "<concise statement of the insight>"
  }},
  {{
    "keypoint_num": 2,
    "keypoint": "<concise statement of the insight>"
  }}
]
""".strip()


def fine_grained_prompt_template(keypoint: str, context: str, prompt: str) -> str:
    """
    Template for generating fine-grained prompts from keypoints.

    Parameters
    ----------
    keypoint : str
        The keypoint to generate a prompt for.
    context : str
        The context/background information for the task.
    prompt : str
        The original prompt/question that the report answers.

    Returns
    -------
    str
        Formatted prompt for LLM.
    """
    return f"""
You are tasked with generating a fine-grained prompt for a functional genomic benchmark.

I will provide you a keypoint, the context, and the original prompt. Please generate a fine-grained prompt for the keypoint.

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

--- BEGIN ORIGINAL PROMPT ---
{prompt}
--- END ORIGINAL PROMPT ---

--- BEGIN KEYPOINT ---
{keypoint}
--- END KEYPOINT ---

Guidelines:
- Avoid asking leading questions. If the keypoint is not explicitly stated in the context or original prompt, design a prompt that could guide someone to arrive at that keypoint through reasoning.
- The prompt shouldn't contain many specific gene names. Try replacing them with pathways, biological processes, or functional categories instead.
- Make the prompt detailed and specific enough to test deep understanding of the keypoint.
- Use phrases like "how does", "what evidence", "which patterns", "how might" to encourage analytical thinking.
- The prompt should be answerable based on the keypoint, context, and original prompt.

Output format (strict JSON):
{{
  "fine_grained_prompt": "<detailed question testing understanding of this keypoint>"
}}
""".strip()


def _sanitize_column_name_for_filename(col: str) -> str:
    """Sanitize a column name for use in a filename."""
    sanitized = re.sub(r"[^\w\s-]", "", col)
    sanitized = re.sub(r"[-\s]+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized if sanitized else "unknown"


def _extract_report_source_from_filename(filepath: str) -> Optional[str]:
    """
    Extract report source name from breakdown JSON filename.

    Examples:
        "insights_breakdown_Discovera_gpt_4o.json" -> "Discovera_gpt_4o"
        "output/insights_breakdown_Biomni_o4_mini.json" -> "Biomni_o4_mini"
        "benchmark_breakdown_new.csv" -> None (CSV files don't have report source)

    Returns:
        str: Report source name, or None if not found/not applicable
    """
    filename = os.path.basename(filepath)

    # Check if it's a JSON breakdown file
    if filename.startswith("insights_breakdown_") and filename.endswith(".json"):
        # Extract the part between "insights_breakdown_" and ".json"
        report_source = filename[len("insights_breakdown_") : -len(".json")]
        return report_source if report_source else None

    return None


def _generate_output_path_for_column(
    report_column: str, output_base_path: Optional[str] = None
) -> str:
    """Generate output path for a single report column."""
    if output_base_path:
        return output_base_path

    sanitized = _sanitize_column_name_for_filename(report_column)
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return f"output/insights_breakdown_{sanitized}.json"


def _process_single_report_column(
    report_column: str,
    tasks_data: List[Dict[str, str]],
    output_path: str,
    llm,
    generate_fine_grained_prompts: bool,
) -> Dict[str, Any]:
    """
    Process a single report column and generate breakdown JSON.

    Returns:
        Dict[str, Any]: JSON structure with task breakdowns
    """
    # Load existing results if output file exists
    json_output = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                json_output = json.load(f)
            print(
                f"[INFO] [{report_column}] Loaded existing results from {output_path}: {len(json_output)} tasks"
            )
        except Exception as e:
            print(
                f"[WARNING] [{report_column}] Failed to load existing results from {output_path}: {e}"
            )
            json_output = {}

    # Process each task
    for task_row in tqdm(tasks_data, desc=f"Processing {report_column}"):
        task_id = task_row.get("ID", "").strip()
        context = task_row.get("Context/Background", "").strip()
        prompt = task_row.get("Prompt", "").strip()

        if not context or not prompt:
            print(
                f"[WARNING] [{report_column}] Task {task_id}: Missing context or prompt, skipping"
            )
            continue

        # Check if task already exists with complete insights
        if task_id in json_output and json_output[task_id].get("insights"):
            existing_insights = json_output[task_id]["insights"]
            # Check if we need fine-grained prompts and if they're missing
            needs_fine_grained = (
                generate_fine_grained_prompts
                and existing_insights
                and "fine-grained prompt" not in existing_insights[0]
            )

            if not needs_fine_grained:
                print(
                    f"[INFO] [{report_column}] Task {task_id}: Using {len(existing_insights)} existing insights, skipping LLM calls"
                )
                continue
            else:
                print(
                    f"[INFO] [{report_column}] Task {task_id}: Found {len(existing_insights)} existing insights "
                    f"but missing fine-grained prompts, will generate them"
                )
                # Use existing insights and add fine-grained prompts
                task_context = json_output[task_id].get("context", context)
                task_prompt = json_output[task_id].get("prompt", prompt)

                # Generate fine-grained prompts for existing insights
                for insight in existing_insights:
                    if "fine-grained prompt" in insight:
                        continue  # Already has fine-grained prompt

                    keypoint = insight.get("insight", "").strip()
                    if not keypoint:
                        continue

                    keypoint_id = insight.get("id")
                    fine_prompt_template = fine_grained_prompt_template(
                        keypoint, task_context, task_prompt
                    )
                    try:
                        fine_prompt_response = llm.run(
                            fine_prompt_template, json_output=True
                        )

                        if isinstance(fine_prompt_response, dict):
                            fine_grained_prompt = fine_prompt_response.get(
                                "fine_grained_prompt", ""
                            ).strip()
                            if fine_grained_prompt:
                                insight["fine-grained prompt"] = fine_grained_prompt
                    except Exception as e:
                        print(
                            f"[ERROR] [{report_column}] Task {task_id}, Keypoint {keypoint_id}: "
                            f"Failed to generate fine-grained prompt: {e}"
                        )

                print(
                    f"[INFO] [{report_column}] Task {task_id}: Added fine-grained prompts to existing insights"
                )
                continue

        # Get report text for this column
        report_text = task_row.get(report_column, "").strip()
        if not report_text:
            print(f"[WARNING] [{report_column}] Task {task_id}: Empty report, skipping")
            continue

        # Step 1: Generate keypoints from report
        keypoints_prompt = breakdown_template(report_text)
        try:
            keypoints_response = llm.run(keypoints_prompt, json_output=True)

            if not isinstance(keypoints_response, list):
                print(
                    f"[WARNING] [{report_column}] Task {task_id}: "
                    f"Expected list for keypoints, got {type(keypoints_response)}"
                )
                continue

            if not keypoints_response:
                print(
                    f"[WARNING] [{report_column}] Task {task_id}: "
                    f"No keypoints generated"
                )
                continue

            # Initialize task structure if not exists
            if task_id not in json_output:
                json_output[task_id] = {
                    "context": context,
                    "prompt": prompt,
                    "insights": [],
                }

            print(
                f"[INFO] [{report_column}] Task {task_id}: "
                f"Generated {len(keypoints_response)} keypoints"
            )

            # Step 2: Generate fine-grained prompt for each keypoint (if enabled)
            for keypoint_item in keypoints_response:
                keypoint_num = keypoint_item.get("keypoint_num")
                keypoint = keypoint_item.get("keypoint", "").strip()

                if not keypoint:
                    continue

                fine_grained_prompt = None

                if generate_fine_grained_prompts:
                    # Generate fine-grained prompt for this keypoint
                    fine_prompt_template = fine_grained_prompt_template(
                        keypoint, context, prompt
                    )
                    try:
                        fine_prompt_response = llm.run(
                            fine_prompt_template, json_output=True
                        )

                        if not isinstance(fine_prompt_response, dict):
                            print(
                                f"[WARNING] [{report_column}] Task {task_id}, "
                                f"Keypoint {keypoint_num}: Expected dict for fine-grained prompt, "
                                f"got {type(fine_prompt_response)}"
                            )
                            continue

                        fine_grained_prompt = fine_prompt_response.get(
                            "fine_grained_prompt", ""
                        ).strip()

                        if not fine_grained_prompt:
                            print(
                                f"[WARNING] [{report_column}] Task {task_id}, "
                                f"Keypoint {keypoint_num}: Empty fine-grained prompt"
                            )
                            continue

                    except Exception as e:
                        print(
                            f"[ERROR] [{report_column}] Task {task_id}, "
                            f"Keypoint {keypoint_num}: Failed to generate fine-grained prompt: {e}"
                        )
                        continue

                # Build insight dictionary
                insight_dict = {
                    "id": keypoint_num,
                    "insight": keypoint,
                }

                if generate_fine_grained_prompts and fine_grained_prompt:
                    insight_dict["fine-grained prompt"] = fine_grained_prompt

                # Add insight to the task
                json_output[task_id]["insights"].append(insight_dict)

            completed_count = len(
                [kp for kp in keypoints_response if kp.get("keypoint")]
            )
            prompt_status = (
                "with prompts" if generate_fine_grained_prompts else "without prompts"
            )
            print(
                f"[INFO] [{report_column}] Task {task_id}: "
                f"Completed {completed_count} keypoints {prompt_status}"
            )

        except Exception as e:
            print(
                f"[ERROR] [{report_column}] Task {task_id}: "
                f"Failed to generate breakdown: {e}"
            )
            continue

    # Sort insights by id within each task
    for task_id in json_output:
        json_output[task_id]["insights"].sort(key=lambda x: x["id"])

    # Write to JSON
    if json_output:
        # Sort task IDs for consistent output
        sorted_task_ids = sorted(
            json_output.keys(),
            key=lambda x: (
                float(x) if x.replace(".", "").isdigit() else float("inf"),
                x,
            ),
        )

        # Create ordered output dictionary
        ordered_output = {task_id: json_output[task_id] for task_id in sorted_task_ids}

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ordered_output, f, indent=2, ensure_ascii=False)

        total_insights = sum(
            len(task_data["insights"]) for task_data in ordered_output.values()
        )
        print(
            f"[INFO] [{report_column}] Wrote breakdown to {output_path}: "
            f"{len(ordered_output)} tasks, {total_insights} insights"
        )

    return json_output


def generate_insights_breakdown_from_reports(
    csv_path: str,
    report_columns: List[str],
    provider: str = "openai",
    model: str = "gpt-5-nano",
    output_path: Optional[str] = None,
    task_filter: Optional[List[str]] = None,
    generate_fine_grained_prompts: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate insights breakdown from reports in benchmark_verified.csv.

    Generates separate JSON files for each report column.

    Args:
        csv_path (str):
            Path to benchmark_verified.csv containing reports.

        report_columns (List[str]):
            List of CSV column names to use as reports (e.g.,
            ["Ground Truth", "Discovera (gpt-4o)", "Biomni (o4-mini)"])

        provider (str, optional):
            The LLM service provider to use (e.g., "openai", "anthropic").
            Defaults to "openai".

        model (str, optional):
            The model name to use for breakdown generation.
            Defaults to "gpt-5-nano".

        output_path (Optional[str], optional):
            Base path for output files. If None, automatically generates filenames
            from each report column. If provided and multiple columns exist,
            will be used as a prefix.
            Defaults to None.

        task_filter (Optional[List[str]], optional):
            If provided, only generate breakdowns for tasks with IDs in this list.
            Defaults to None (process all tasks).

        generate_fine_grained_prompts (bool, optional):
            If True, generate fine-grained prompts for each insight.
            If False, only generate keypoints without fine-grained prompts.
            Defaults to True.

    Returns:
        Dict[str, Dict[str, Any]]:
            A dictionary mapping report column names to their JSON breakdowns.
            Each breakdown is a dictionary with task IDs as keys, containing:
            - "context": str - Context text
            - "prompt": str - Original prompt
            - "insights": List[Dict] - List of insights, each with:
                - "id": int - Insight ID
                - "insight": str - Insight text
                - "fine-grained prompt": str - Fine-grained prompt (if generate_fine_grained_prompts=True)
    """
    llm = get_llm(provider, model, api_key=load_openai_key())

    # Load CSV data
    print(f"[INFO] Loading reports from {csv_path}...")
    tasks_data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() if h else h for h in reader.fieldnames]
        for row in reader:
            task_id = row.get("ID", "").strip()
            if not task_id:
                continue
            if task_filter and task_id not in task_filter:
                continue
            tasks_data.append(row)

    print(f"[INFO] Found {len(tasks_data)} tasks to process")
    print(f"[INFO] Processing {len(report_columns)} report columns: {report_columns}")
    print(f"[INFO] Generate fine-grained prompts: {generate_fine_grained_prompts}")

    start_time = time.time()
    all_outputs = {}

    # Process each report column separately
    for report_column in report_columns:
        # Generate output path for this column
        if output_path:
            # If base path provided, use its directory but generate filename from column
            base_dir = os.path.dirname(output_path) or "output"
            sanitized_col = _sanitize_column_name_for_filename(report_column)
            column_output_path = os.path.join(
                base_dir, f"insights_breakdown_{sanitized_col}.json"
            )
        else:
            # Auto-generate path from column name
            column_output_path = _generate_output_path_for_column(
                report_column, output_path
            )

        print(f"\n{'='*60}")
        print(f"[INFO] Processing report column: {report_column}")
        print(f"[INFO] Output file: {column_output_path}")
        print(f"{'='*60}")

        # Process this column
        column_output = _process_single_report_column(
            report_column=report_column,
            tasks_data=tasks_data,
            output_path=column_output_path,
            llm=llm,
            generate_fine_grained_prompts=generate_fine_grained_prompts,
        )

        all_outputs[report_column] = column_output

    elapsed = time.time() - start_time
    total_tasks = sum(len(output) for output in all_outputs.values())
    total_insights = sum(
        sum(len(task_data["insights"]) for task_data in output.values())
        for output in all_outputs.values()
    )
    print(f"\n{'='*60}")
    print(
        f"[INFO] Completed all columns in {elapsed:.2f} seconds. "
        f"Total tasks: {total_tasks}, Total insights: {total_insights}"
    )
    print(f"{'='*60}")

    return all_outputs


def generate_questions_from_breakdown(
    breakdown_path: str,
    provider: str = "openai",
    model: str = "gpt-5-nano",
    output_path: Optional[str] = None,
    task_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice questions from benchmark breakdown (CSV or JSON).
    Generates one MCQ per keypoint, batched by task (context + list of keypoints).

    This function loads the benchmark breakdown (CSV or JSON), groups keypoints by task_id,
    and generates questions in batches using the Context and all keypoints for
    each task.

    Args:
        breakdown_path (str):
            Path to the benchmark breakdown file (CSV or JSON).
            Format is auto-detected based on file extension.

        provider (str, optional):
            The LLM service provider to use (e.g., "openai", "anthropic").
            Defaults to "openai".

        model (str, optional):
            The model name to use for question generation.
            Defaults to "gpt-5-nano".

        output_path (Optional[str], optional):
            If provided, the generated questions will also be written to this
            path as a JSON file. Defaults to None.

        task_filter (Optional[List[str]], optional):
            If provided, only generate questions for tasks with IDs in this list.
            Defaults to None (process all tasks).

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries, each containing:
            - "task_id" (str): Task ID from the breakdown
            - "question_id" (str): Keypoint number as string
            - "question_source" (str): "groundtruth"
            - "question" (str): The generated question
            - "choices" (List[str]): Answer choices for the question
            - "answer" (str): The correct answer (A, B, C, or D)
            - "ground_truth_keypoint" (str): The ground truth keypoint text

    Raises:
        FileNotFoundError: If the breakdown file does not exist.
        ValueError: If the file format is not supported.
        RuntimeError: If the LLM call fails or response parsing fails.
    """
    # Auto-detect file format and load data
    file_ext = os.path.splitext(breakdown_path)[1].lower()

    # Extract report source from filename (for JSON files)
    report_source = _extract_report_source_from_filename(breakdown_path)

    if file_ext == ".json":
        print(f"[INFO] Loading breakdown from JSON: {breakdown_path}")
        if report_source:
            print(f"[INFO] Detected report source: {report_source}")
        tasks = load_benchmark_breakdown_json(breakdown_path)
    elif file_ext == ".csv":
        print(f"[INFO] Loading breakdown from CSV: {breakdown_path}")
        tasks = load_benchmark_breakdown(breakdown_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. " f"Supported formats: .csv, .json"
        )

    if not tasks:
        print(f"[WARNING] No valid tasks found in {breakdown_path}")
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
                    "question_source": (
                        report_source
                        if report_source
                        else q.get("question_source", "groundtruth")
                    ),
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

                # Build filename with report source if available
                if report_source:
                    filename = f"qs{num_questions}_{report_source}_{provider}_{model_name}.json"
                else:
                    filename = f"qs{num_questions}_{provider}_{model_name}.json"

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

        # Get CSV row for this task
        csv_row = csv_data.get(task_id)
        if not csv_row:
            print(f"[WARNING] Task {task_id} not found in CSV")
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

            # Extract report source from question file or questions
            question_report_source = None
            if questions and "question_source" in questions[0]:
                question_report_source = questions[0].get("question_source")
            else:
                # Try to extract from filename (e.g., "qs10_Discovera_gpt_4o_openai_gpt_5_nano.json")
                # Pattern: qs{num}_{report_source}_{provider}_{model}.json
                import re

                match = re.match(r"qs\d+_(.+?)_(openai|anthropic)_", question_file)
                if match:
                    question_report_source = match.group(1)

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
                        result_dict = {
                            "task_id": q.get("task_id", task_id),
                            "question_id": q.get("question_id", ""),
                            "question": q.get("question", ""),
                            "question_source": q.get(
                                "question_source", question_report_source
                            ),
                            "choices": q.get("choices", []),
                            "answer": q.get("answer", ""),
                            "prediction": response.get("answer", None),
                            "confidence": response.get("confidence", None),
                            "report_source": source_name,
                        }
                        results.append(result_dict)
                    except Exception as e:
                        print(
                            f"[ERROR] Failed to answer question for task "
                            f"{task_id}: {e}"
                        )
                        error_dict = {
                            "task_id": q.get("task_id", task_id),
                            "question_id": q.get("question_id", ""),
                            "question": q.get("question", ""),
                            "question_source": q.get(
                                "question_source", question_report_source
                            ),
                            "choices": q.get("choices", []),
                            "answer": q.get("answer", ""),
                            "prediction": None,
                            "confidence": None,
                            "report_source": source_name,
                            "error": str(e),
                        }
                        results.append(error_dict)

                # Save results
                if results:
                    task_answers_dir = os.path.join(answers_dir, task_id)
                    os.makedirs(task_answers_dir, exist_ok=True)

                    # Extract number of questions from question filename
                    # e.g., "qs12_Discovera_gpt_4o_openai_gpt_5_nano.json" -> "qs12"
                    qs_match = question_file.split("_")[0]  # "qs12"
                    num_questions = (
                        qs_match.replace("qs", "")
                        if qs_match.startswith("qs")
                        else str(len(results))
                    )

                    # Format model name
                    model_name = model.replace("/", "-").replace("-", "_")

                    # Create answer filename with report sources
                    # Use question report source if available, otherwise use answer report source
                    if question_report_source:
                        # Include both question report source and answer report source
                        answer_filename = f"ans{num_questions}_{question_report_source}_rs{source_name}_{provider}_{model_name}.json"
                    else:
                        # Fallback to old format
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
                        print(f"[WARNING] Failed to load report {report_path}: {e}")
                else:
                    print(f"[WARNING] Fine-grained report not found: {report_path}")

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
                            "question_source": q.get("question_source", "groundtruth"),
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
                            "question_source": q.get("question_source", "groundtruth"),
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
                            model = "_".join(parts[i + 2 :])  # Join remaining parts
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
            coverage_prompt = check_coverage_prompt(keypoint, report_text, report_name)

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
