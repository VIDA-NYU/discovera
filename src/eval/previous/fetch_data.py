"""
main.py

This script automatically downloads all data files associated with the ground truth.

It reads a CSV file (benchmark.csv) containing dataset information, extracts URLs from
the "Data Set " column, and downloads each file to a designated folder.

Features:
- Downloads generic HTTP/HTTPS files.
- Downloads Google Drive files using gdown.
- Automatically generates filenames based on a unique "id" for each row.
- Ensures the destination folder exists.
- Can be run directly as a standalone script.

Usage:
    python main.py
"""

import os
import re
import pandas as pd
import requests
import gdown

# ---------------------- Configuration ----------------------
DATA_FOLDER = "../data/benchmark/"         
CSV_FILE = os.path.join(DATA_FOLDER, "benchmark.csv") 
URL_PATTERN = r'https?://[^\s]+'            # Regex pattern to extract URLs
# -----------------------------------------------------------

def download_file(url, filename, dest_folder=DATA_FOLDER):
    """
    Download a generic file via HTTP/HTTPS.

    Args:
        url (str): URL of the file to download.
        filename (str): Name to save the file as.
        dest_folder (str): Folder to save the file.

    Returns:
        str: Full path to the downloaded file.
    """
    path = os.path.join(dest_folder, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return path

def download_gdrive(url, filename, dest_folder=DATA_FOLDER):
    """
    Download a file from Google Drive using gdown.

    Args:
        url (str): Google Drive share URL.
        filename (str): Name to save the file as.
        dest_folder (str): Folder to save the file.

    Returns:
        str or None: Full path to the downloaded file, or None if download failed.
    """
    file_id = None
    if "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    elif "/d/" in url:
        file_id = url.split("/d/")[1].split("/")[0]

    if file_id:
        path = os.path.join(dest_folder, filename)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
        return path
    return None

def main():
    """
    Main function to download all files referenced in the CSV.

    - Reads the CSV file.
    - Creates a unique "id" for each row.
    - Extracts URLs from the "Data Set " column.
    - Downloads each file, naming them based on the unique ID.
    """
    df = pd.read_csv(CSV_FILE)

    df["id"] = [f"gt_{i}" for i in range(len(df))]

    # Ensure the destination folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    for idx, text in df["Data Set "].items():  # Keep exact column name
        if pd.isna(text):
            continue

        urls = re.findall(URL_PATTERN, str(text))
        if not urls:
            continue

        for link_idx, url in enumerate(urls, start=1):
            file_id = df.at[idx, "id"]
            filename = f"{file_id}_link{link_idx}.xlsx"

            try:
                if "drive.google.com" in url:
                    path = download_gdrive(url, filename)
                else:
                    path = download_file(url, filename)
                print(f"Downloaded {url} -> {path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    main()
