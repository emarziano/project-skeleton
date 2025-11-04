import os
import shutil
import requests
from zipfile import ZipFile
from io import BytesIO


def prepare_tinyimagenet(dataset_root: str = "dataset"):
    """
    Scarica e prepara Tiny-ImageNet nel percorso specificato.
    """
    zip_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    extract_dir = os.path.join(dataset_root, "tiny-imagenet-200")

    # Se già esiste, salta il download
    if os.path.exists(extract_dir):
        print(f"✅ Dataset already found at {extract_dir}, skipping download.")
    else:
        os.makedirs(dataset_root, exist_ok=True)
        print("⬇️ Downloading Tiny-ImageNet...")
        response = requests.get(zip_url)
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(dataset_root)
        print("✅ Extraction complete.")

    # Percorsi rilevanti
    val_dir = os.path.join(extract_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    # Se le immagini sono già state spostate, non ripetere
    if not os.path.exists(images_dir):
        print("ℹ️ Validation folder already restructured, skipping.")
        return

    print("📂 Restructuring validation folder...")

    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Cannot find {annotations_file}")

    with open(annotations_file, "r") as f:
        for line in f:
            filename, cls, *_ = line.strip().split("\t")
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_dir, filename)
            dst = os.path.join(cls_dir, filename)

            # Verifica che il file esista
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print(f"⚠️ Missing image: {src}, skipping...")

    # Rimuovi la vecchia directory "images" se esiste
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print("✅ Validation folder restructured successfully.")
