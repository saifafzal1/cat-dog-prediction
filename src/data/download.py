"""
Dataset download module for Cats vs Dogs classification.

This module handles downloading the Kaggle Cats and Dogs dataset.
The dataset can be downloaded either via Kaggle API or manual download.
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_dataset_kaggle(output_dir: str) -> None:
    """
    Download the Cats and Dogs dataset using Kaggle API.

    Prerequisites:
        - Kaggle API credentials must be configured
        - kaggle.json should be in ~/.kaggle/ directory

    Args:
        output_dir: Directory where the dataset will be downloaded

    Raises:
        ImportError: If kaggle package is not installed
        Exception: If download fails
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Install with: pip install kaggle"
        )

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Create output directory if it does not exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Cats and Dogs dataset from Kaggle...")

    # Download the dataset
    # Dataset: microsoft/cats-vs-dogs or similar Kaggle dataset
    api.dataset_download_files(
        "karakaggle/kaggle-cat-vs-dog-dataset",
        path=output_dir,
        unzip=True
    )

    logger.info(f"Dataset downloaded successfully to {output_dir}")


def extract_dataset(zip_path: str, output_dir: str) -> None:
    """
    Extract a zipped dataset to the specified directory.

    Args:
        zip_path: Path to the zip file
        output_dir: Directory where contents will be extracted

    Raises:
        FileNotFoundError: If zip file does not exist
        zipfile.BadZipFile: If the file is not a valid zip file
    """
    zip_path = Path(zip_path)
    output_path = Path(output_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} to {output_dir}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    logger.info("Extraction completed successfully")


def organize_dataset(raw_dir: str) -> None:
    """
    Organize the raw dataset into a standard structure.

    Expected output structure:
        raw_dir/
            cats/
                cat.0.jpg
                cat.1.jpg
                ...
            dogs/
                dog.0.jpg
                dog.1.jpg
                ...

    Args:
        raw_dir: Directory containing the raw downloaded data
    """
    raw_path = Path(raw_dir)

    # Create category directories
    cats_dir = raw_path / "cats"
    dogs_dir = raw_path / "dogs"
    cats_dir.mkdir(exist_ok=True)
    dogs_dir.mkdir(exist_ok=True)

    # Find all image files and organize them
    image_extensions = {".jpg", ".jpeg", ".png"}

    for file_path in raw_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            filename = file_path.name.lower()

            # Determine category based on filename
            if "cat" in filename:
                destination = cats_dir / file_path.name
            elif "dog" in filename:
                destination = dogs_dir / file_path.name
            else:
                continue

            # Move file if not already in the correct directory
            if file_path.parent != destination.parent:
                shutil.copy2(file_path, destination)

    logger.info(f"Dataset organized into {cats_dir} and {dogs_dir}")

    # Count images in each category
    cat_count = len(list(cats_dir.glob("*")))
    dog_count = len(list(dogs_dir.glob("*")))
    logger.info(f"Found {cat_count} cat images and {dog_count} dog images")


def setup_dataset(output_dir: str, kaggle_download: bool = True) -> None:
    """
    Main function to set up the dataset.

    This function orchestrates the download and organization of the dataset.

    Args:
        output_dir: Base directory for the raw data
        kaggle_download: If True, download from Kaggle. If False, assume
                        manual download and only organize.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if kaggle_download:
        download_dataset_kaggle(output_dir)

    organize_dataset(output_dir)

    logger.info("Dataset setup completed")


if __name__ == "__main__":
    # Default execution: download and setup dataset
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.config import load_config, get_data_config

    config = load_config()
    data_config = get_data_config(config)

    setup_dataset(
        output_dir=data_config.get("raw_dir", "data/raw"),
        kaggle_download=True
    )
