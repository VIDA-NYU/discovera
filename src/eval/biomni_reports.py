#!/usr/bin/env python3
import os
import shutil
import zipfile
import pandas as pd
import time
import tomllib 
import argparse

from gradio_client import Client, handle_file


def load_biomni_credentials(conf_path=os.path.expanduser("../../.beaker.conf")):
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"beaker.conf not found: {conf_path}")

    with open(conf_path, "rb") as f:
        config = tomllib.load(f)

    biomni_section = config.get("biomni", {})
    username = biomni_section.get("username")
    password = biomni_section.get("password")

    if not username or not password:
        raise ValueError("Missing username or password under [biomni] in beaker.conf")

    # Strip quotes if user added them
    username = username.strip().strip('"').strip("'")
    password = password.strip().strip('"').strip("'")

    return username, password


# --- Utility Functions ---
def restart_session():
    return Client("https://app.biomni.stanford.edu/app/")


def biomni_login(client):
    USERNAME, PASSWORD = load_biomni_credentials()

    login_result = client.predict(
        username=USERNAME,
        password=PASSWORD,
        api_name="/handle_login"
    )
    print(f"Logged in as: {USERNAME}")
    return login_result[5]   # token


def biomni_logout():
    client = Client("https://app.biomni.stanford.edu/app/")
    return client.predict(api_name="/handle_logout")


def extract_final_answer(task_result):
    if isinstance(task_result, tuple):
        task_result = list(task_result)
    direct_response_fallback = None

    for item in task_result:
        messages = item if isinstance(item, list) else [item]
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                metadata = msg.get("metadata")
                if metadata and metadata.get("title") == "✅ Answer":
                    return msg.get("content")
                if metadata and metadata.get("title") == "💬 Direct Response":
                    direct_response_fallback = msg.get("content")

    return direct_response_fallback


def download_and_save_session_log(client, session_id, permanent_folder):
    os.makedirs(permanent_folder, exist_ok=True)
    log_file = client.predict(
        selected_session_id=session_id,
        api_name="/handle_log_download"
    )

    temp_path = log_file.get("value")
    destination_path = os.path.join(permanent_folder, os.path.basename(temp_path))
    shutil.copy(temp_path, destination_path)

    if destination_path.endswith(".zip"):
        extract_folder = os.path.join(
            permanent_folder,
            os.path.splitext(os.path.basename(destination_path))[0]
        )
        os.makedirs(extract_folder, exist_ok=True)
        with zipfile.ZipFile(destination_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        return extract_folder

    return destination_path


def get_task_files(task_id, tasks_folder):
    task_path = os.path.join(tasks_folder, str(task_id))
    if not os.path.exists(task_path):
        return []
    return [
        os.path.join(task_path, f)
        for f in os.listdir(task_path)
        if os.path.isfile(os.path.join(task_path, f)) and not f.startswith(".")
    ]


# -----------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------
def process_biomni_dataframe(df, tasks_folder, permanent_folder_logs, output_csv_path, model="OpenAI O4-Mini"):

    results = []

    for idx, row in df.iterrows():
        start_time = time.time()

        task_id_raw = row.get("ID", idx)
        task_id = str(int(task_id_raw)) if isinstance(task_id_raw, float) and task_id_raw.is_integer() else str(task_id_raw)

        input_text = f"{row['Context/Background']} {row['Prompt']} {row['Prompt Restriction']}"
        print(f"Processing task {task_id}: {input_text[:50]}...")

        files = [handle_file(f) for f in get_task_files(task_id, tasks_folder)]

        client = restart_session()
        biomni_login(client)

        task_result = client.predict(
            input_value={"text": input_text, "files": files},
            inner_history=[],
            main_history=[],
            model=model,
            direct_mode=False,
            api_name="/process_input"
        )

        sessions = client.predict(api_name="/_populate_session_dropdown")
        latest_session_id = sessions["choices"][0][1] if sessions.get("choices") else None

        log_path = download_and_save_session_log(client, latest_session_id, permanent_folder_logs) if latest_session_id else None
        final_answer = extract_final_answer(task_result)
        processing_time = time.time() - start_time

        results.append({
            "task_id": task_id,
            "session_id": latest_session_id,
            "log_path": log_path,
            "task_result": task_result,
            "final_answer": final_answer,
            "model": model,
            "processing_time_secs": processing_time,
        })

        # Save progress after each row
        pd.DataFrame(results).to_csv(output_csv_path, index=False)
        print(f"Saved progress to {output_csv_path}")

        biomni_logout()

    return pd.DataFrame(results)


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="OpenAI O4-Mini")
    parser.add_argument("--experiment", default=f"biomni_run_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated list of task IDs to process. Example: --task_ids 1,5,22"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.benchmark)
    # Normalize task ID column to string for matching
    df["ID_str"] = df["ID"].apply(
        lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
    )

    # If user passed task IDs, filter for those only
    if args.task_ids:
        selected_ids = [x.strip() for x in args.task_ids.split(",")]
        print(f"Running only on task IDs: {selected_ids}")
        df = df[df["ID_str"].isin(selected_ids)]
        
    output_df = process_biomni_dataframe(
        df=df,
        tasks_folder=args.tasks,
        permanent_folder_logs=args.logs,
        output_csv_path=args.output,
        model=args.model
    )

    print("Processing complete. Final results saved.")

