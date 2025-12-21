import os
import requests
import zipfile
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TEMP_ZIP_PATH = ROOT_DIR / "temp_data.zip"

DROPBOX_BASEURL = "https://www.dropbox.com/s"

DROPBOX_RESOURCES = {
    "train_model": f"{DROPBOX_BASEURL}/nifi5nj1oj0fu2i/data.zip?dl=1",
    "other_embeddings":f"{DROPBOX_BASEURL}/tzkaoagzxuxtwqs/data.zip?dl=1",
    "distorted_smallNYT": f"{DROPBOX_BASEURL}/6q5jhhmxdmc8n1e/data.zip?dl=1"
}

def _data_loaded() -> bool:
    data_file_count = len(list(DATA_DIR.glob('*.mat')))
    return data_file_count >= 211

def _download_file(url: str, save_path: Path):
    print(f"  ⬇️ Starting download from: {url}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"  🎉 Download successful! File saved to {save_path.name}")

def _unzip_and_clean(zip_path: Path, extract_dir: Path):
    print(f"  ⚙️ Unzipping {zip_path.name}...")
    temp_extract_dir = zip_path.with_suffix('.temp_extract')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_extract_dir.mkdir(exist_ok=True)
        zip_ref.extractall(temp_extract_dir)

    content_folder = list(temp_extract_dir.iterdir())[0]

    for item in content_folder.iterdir():
        shutil.move(str(item), extract_dir / item.name)
    print(f"  ✅ Content moved successfully to {extract_dir.name}")

    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
    if zip_path.exists():
        os.remove(zip_path)
    print(f"  🧹 Cleaned up temporary files")


def load_data():
    """
    Iterates through all DROPBOX_RESOURCES, downloads them, extracts them to the
    correct directory (../data), and removes the temporary zip file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _data_loaded():
        print("Data seems to be loaded already. Skipping download.")
        return

    print("--- Starting Data Loading Process ---")
    for name, url in DROPBOX_RESOURCES.items():
        print(f"\nProcessing resource: **{name}**")
        # Download the ZIP file
        _download_file(url, TEMP_ZIP_PATH)
        # Unzip the content and move files to data directory
        _unzip_and_clean(TEMP_ZIP_PATH, DATA_DIR)

    print("\n--- Data Loading Process Complete ---")

if __name__ == "__main__":
    print(f"Target data directory: {DATA_DIR.resolve()}")
    load_data()