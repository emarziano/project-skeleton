# utils/download_dataset.py
import os
import requests
from zipfile import ZipFile
from io import BytesIO

def download_file(url: str, output_dir: str = "dataset", overwrite: bool = False):
    """
    Download a file from the given URL and extract it if it's a ZIP archive.

    Args:
        url (str): URL of the file to download.
        output_dir (str): Directory where to extract or save the data.
        overwrite (bool): If True, re-download even if data already exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataset already exists to avoid redundant downloads
    dataset_name = url.split("/")[-1].replace(".zip", "")
    dataset_path = os.path.join(output_dir, dataset_name)

    if os.path.exists(dataset_path) and not overwrite:
        print(f"✅ Dataset already found at {dataset_path}, skipping download.")
        return dataset_path

    print(f"⬇️  Downloading dataset from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # stop on HTTP errors

    if url.endswith(".zip"):
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_dir)
        print(f"✅ Extracted dataset to {output_dir}")
    else:
        file_path = os.path.join(output_dir, os.path.basename(url))
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved dataset file to {file_path}")

    return dataset_path
