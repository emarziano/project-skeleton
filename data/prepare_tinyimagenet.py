import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_prepare_tinyimagenet(data_dir: str = "data/tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract
    print("Downloading Tiny-ImageNet dataset...")
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("data")
    print("Extraction complete.")

    # Fix validation folder structure
    val_dir = os.path.join(data_dir, "val")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    with open(annotations_file) as f:
        for line in f:
            fn, cls, *_ = line.split("\t")
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            shutil.copyfile(
                os.path.join(val_dir, "images", fn),
                os.path.join(val_dir, cls, fn)
            )
    shutil.rmtree(os.path.join(val_dir, "images"))
    print("Validation folder restructured.")
