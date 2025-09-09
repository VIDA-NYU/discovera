#!/usr/bin/env python3
"""
Baseline LLM Experimental Setup for Biomedical Benchmarking

This script implements the baseline workflow for evaluating an LLM (e.g., GPT-4o)
on biomedical tasks using context and prompt text from a benchmark CSV.

Key features:
1. Loads benchmark CSV containing 'Context/Background' and 'Prompt'.
2. Drops any rows with missing prompts (NaN).
3. For each task:
   - Finds associated files in the data directory (if any).
   - Splits large files (>5MB) into smaller parts for upload.
   - Creates a vector store and uploads files (if files exist).
   - Queries the LLM using context + prompt, optionally with vector store search.
4. Collects LLM outputs and full response metadata.
5. Saves results to a CSV file for evaluation.

This baseline script serves as a reference setup (Control / Baseline) for
future experimental comparisons.
"""

import os
import sys
import time
import pandas as pd
from openai import OpenAI

# Add relative module path to import custom prompting functions
module_dir = "../"  # adjust relative path if needed
sys.path.append(os.path.abspath(module_dir))
from src.eval.prompting import *  

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB maximum file size for upload
MAX_RETRIES = 3  # Maximum retries for file upload

def split_large_file(file_path, max_size=MAX_FILE_SIZE):
    """
    Split a file into smaller parts if it exceeds the maximum size.

    Args:
        file_path (str): Path to the input file.
        max_size (int): Maximum size in bytes for each split part.

    Returns:
        list[str]: List of file paths (original or split) to upload.
    """
    files_to_upload = []
    if os.path.getsize(file_path) <= max_size:
        return [file_path]

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_lines, current_size, part_num = [], 0, 1
    base_name = os.path.basename(file_path)
    data_path = os.path.dirname(file_path)

    for line in lines:
        line_size = len(line.encode("utf-8"))
        if current_size + line_size > max_size:
            split_path = os.path.join(data_path, f"{base_name}_part{part_num}.txt")
            with open(split_path, "w", encoding="utf-8") as sf:
                sf.writelines(current_lines)
            files_to_upload.append(split_path)
            current_lines, current_size = [], 0
            part_num += 1
        current_lines.append(line)
        current_size += line_size

    if current_lines:
        split_path = os.path.join(data_path, f"{base_name}_part{part_num}.txt")
        with open(split_path, "w", encoding="utf-8") as sf:
            sf.writelines(current_lines)
        files_to_upload.append(split_path)

    return files_to_upload

def upload_files_with_retry(client, vector_store_id, files):
    """
    Upload files to a vector store with exponential backoff retries.

    Args:
        client (OpenAI): OpenAI client instance.
        vector_store_id (str): Vector store ID for file upload.
        files (list[str]): List of file paths to upload.
    """
    for f_path in files:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with open(f_path, "rb") as f_obj:
                    client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store_id,
                        files=[f_obj]
                    )
                break
            except Exception as e:
                print(f"Attempt {attempt} failed for {f_path}: {e}")
                time.sleep(2 ** attempt)
        else:
            print(f"Failed to upload {f_path} after {MAX_RETRIES} attempts")

def query_llm(client, instructions, input_text, vector_store_id, model_name):
    """
    Query the LLM using the specified model, instructions, and input text.

    Optionally uses a vector store for file-based search if vector_store_id is provided.

    Args:
        client (OpenAI): OpenAI client instance.
        instructions (str): LLM instructions/context.
        input_text (str): Input combining context and prompt.
        vector_store_id (str or None): Vector store ID for file search, if available.
        model_name (str): Model name (e.g., "gpt-4o").

    Returns:
        tuple[str, str]: (LLM output text, full response metadata as string)
    """
    try:
        tools = [
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ]
        if vector_store_id:
            tools.insert(0, {"type": "file_search", "vector_store_ids": [vector_store_id]})

        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=input_text,
            tools=tools
        )

        text_outputs, full_contents = [], []
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for content_item in item.content:
                    content_dict = content_item.to_dict() if hasattr(content_item, "to_dict") else vars(content_item)
                    text = getattr(content_item, "text", None)
                    if text:
                        text_outputs.append(text)
                    full_contents.append(str(content_dict))

        return "\n\n".join(text_outputs), "\n\n".join(full_contents)
    except Exception as e:
        return f"Failed to get response: {e}", f"Failed to get response: {e}"

def main():
    """Main function for the baseline LLM workflow."""
    # Load benchmark CSV and drop rows with missing prompts
    df = pd.read_csv("../data/benchmark/benchmark.csv")
    df = df.dropna(subset=['Prompt'])
    df['id'] = [f"gt{i}" for i in range(len(df))]

    # Initialize OpenAI client
    client = OpenAI(api_key=load_openai_key(beaker_conf_path="../.beaker.conf"))
    data_path = "../data/benchmark/"
    instructions = "You are Biomedical Researcher."
    model_name = "gpt-4o" #"gpt-5"
    output_rows = []

    for _, row in df.iterrows():
        task_id = row['id']
        print(f"Processing {task_id}...")
        input_text = f"{row['Context/Background']}\n\n{row['Prompt']}"

        # Find matching files for this task
        matching_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith(str(task_id))]
        files_to_upload = []
        vs = None  # Vector store placeholder

        if matching_files:
            # Split large files
            for f_path in matching_files:
                files_to_upload.extend(split_large_file(f_path))

            # Create vector store
            vs_name = f"Baseline - {task_id}"
            print(f"Creating vector store: {vs_name}")
            vs = client.vector_stores.create(name=vs_name)

            # Upload files
            upload_files_with_retry(client, vs.id, files_to_upload)
        else:
            print(f"No matching files for {task_id}. Running LLM with context and prompt only.")
            vs_name = ""

        # Query the LLM
        vector_store_id = vs.id if vs else None
        llm_text, full_response = query_llm(client, instructions, input_text, vector_store_id, model_name)

        # Append results
        output_rows.append({
            "id": task_id,
            "LLM Report Column": llm_text,
            "Full Response Column": full_response,
            "Model Name": model_name,
            "Vector Store ID": vector_store_id or "",
            "Vector Store Name": vs_name
        })

    # Save all outputs to CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv("../data/llm_generated_reports.csv", index=False)
    print("LLM reports saved!")

if __name__ == "__main__":
    main()

