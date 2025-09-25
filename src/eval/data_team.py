import os
import argparse
import pandas as pd
from typing import Optional


def process_experiment_data(
        experiment_id: str,
        base_path: str = "data/experiments",
        output_dir: str = "data/experiments",
        verbose: bool = True
        ) -> str:
    """
    Reads all tabular files (JSON) from the 'answers' subfolder (including subdirectories)
    of a given experiment ID, adds a 'source_file' column, concatenates them, 
    and writes the result to a CSV file.

    Parameters:
        experiment_id (str): The experiment folder name (e.g. 'gpt-5_20250923_173207').
        base_path (str): Base directory containing experiment folders.
        output_dir (Optional[str]): Directory to write the output CSV to. Defaults to base_path.
        verbose (bool): Whether to print processing logs.

    Returns:
        str: Path to the saved combined CSV file.
    """
    questions_path = os.path.join(base_path, experiment_id, "answers")
    output_path = os.path.join(output_dir, experiment_id, "analysis")
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, f"{experiment_id}.csv")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Directory not found: {questions_path}")

    df_list = []
    supported_exts = ['.json']

    # Walk through all subdirectories of questions_path
    for root, _, files in os.walk(questions_path):
        for filename in files:
            if not any(filename.endswith(ext) for ext in supported_exts):
                continue

            file_path = os.path.join(root, filename)

            try:
                df = pd.read_json(file_path)
                df['source_file'] = os.path.relpath(file_path, questions_path)
                df_list.append(df)

                if verbose:
                    print(f"Loaded: {file_path} ({df.shape[0]} rows)")

            except Exception as e:
                if verbose:
                    print(f"Failed to read {file_path}: {e}")

    if not df_list:
        raise ValueError("No valid files found to process.")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

    if verbose:
        print(f"\nCombined data saved to: {output_file} ({combined_df.shape[0]} rows)")

    return output_file



def main():
    parser = argparse.ArgumentParser(description="Process experiment data into a single CSV.")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment ID (e.g. gpt-5_20250923_173207)")
    parser.add_argument("--base-path", "-b", default="data/experiments", help="Base path to experiments")
    parser.add_argument("--output-dir", "-o", default="data/experiments", help="Directory to save combined CSV")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output logs")

    args = parser.parse_args()

    try:
        process_experiment_data(
            experiment_id=args.experiment,
            base_path=args.base_path,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
    except Exception as err:
        print(f"Error: {err}")


if __name__ == "__main__":
    main()
