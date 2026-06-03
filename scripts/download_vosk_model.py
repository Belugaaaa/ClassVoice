from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_MODEL = "vosk-model-small-cn-0.22"
DEFAULT_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_MODEL}.zip"


def download_model(url: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / Path(url).name

    print(f"Downloading {url}")
    print(f"Target: {zip_path}")
    urlretrieve(url, zip_path)

    print("Extracting model...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)

    model_dir = target_dir / zip_path.stem
    print(f"Done: {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Vosk speech model.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Vosk model zip URL.")
    parser.add_argument("--target-dir", default="models", help="Directory to store model files.")
    args = parser.parse_args()

    download_model(args.url, Path(args.target_dir))


if __name__ == "__main__":
    main()
