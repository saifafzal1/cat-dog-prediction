"""
Data preprocessing module for Cats vs Dogs classification.

This module handles image preprocessing, data augmentation, and
splitting the dataset into train/validation/test sets.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_image(image_path: str) -> bool:
    """
    Validate that an image file can be opened and is not corrupted.

    Args:
        image_path: Path to the image file

    Returns:
        True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Verify the image can be loaded
            img.verify()
        # Re-open to check if it can be read (verify() may not catch all issues)
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False


def resize_image(
    image_path: str,
    output_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Resize an image to the target size and save it.

    The image is resized using high-quality Lanczos resampling.
    Images are converted to RGB format to ensure 3 channels.

    Args:
        image_path: Path to the source image
        output_path: Path where the resized image will be saved
        target_size: Target dimensions as (width, height)

    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (handles grayscale and RGBA images)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize using Lanczos resampling for high quality
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

            # Save the resized image
            img_resized.save(output_path, "JPEG", quality=95)

        return True
    except Exception as e:
        logger.error(f"Failed to resize {image_path}: {e}")
        return False


def split_dataset(
    image_paths: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of image paths into train, validation, and test sets.

    The split is performed randomly but reproducibly using the random seed.

    Args:
        image_paths: List of image file paths
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
        test_ratio: Proportion of data for testing (default 0.1)
        random_seed: Seed for random number generator (default 42)

    Returns:
        Tuple containing (train_paths, val_paths, test_paths)

    Raises:
        ValueError: If ratios do not sum to 1.0
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle the paths
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)

    # Calculate split indices
    n_total = len(shuffled_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split the data
    train_paths = shuffled_paths[:n_train]
    val_paths = shuffled_paths[n_train:n_train + n_val]
    test_paths = shuffled_paths[n_train + n_val:]

    return train_paths, val_paths, test_paths


def preprocess_dataset(
    raw_dir: str,
    processed_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Preprocess the entire dataset: validate, resize, and split.

    This function performs the following steps:
    1. Validates all images in the raw directory
    2. Resizes valid images to the target size
    3. Splits images into train/val/test sets
    4. Organizes processed images into the appropriate directories

    Output structure:
        processed_dir/
            train/
                cats/
                dogs/
            val/
                cats/
                dogs/
            test/
                cats/
                dogs/

    Args:
        raw_dir: Directory containing raw images organized by class
        processed_dir: Directory where processed images will be saved
        target_size: Target image dimensions (width, height)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducible splits

    Returns:
        Dictionary with counts for each split and class
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Class labels (subdirectory names)
    classes = ["cats", "dogs"]

    # Create output directory structure
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in classes:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Statistics dictionary
    stats = {}

    for cls in classes:
        class_dir = raw_path / cls

        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        # Get all image paths for this class
        image_extensions = {".jpg", ".jpeg", ".png"}
        all_images = [
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        # Validate images
        valid_images = [p for p in all_images if validate_image(p)]
        logger.info(
            f"Class {cls}: {len(valid_images)}/{len(all_images)} valid images"
        )

        # Split the dataset
        train_imgs, val_imgs, test_imgs = split_dataset(
            valid_images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )

        # Process and save images for each split
        split_data = [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs)
        ]

        for split_name, img_paths in split_data:
            output_dir = processed_path / split_name / cls
            processed_count = 0

            for img_path in img_paths:
                img_name = Path(img_path).name
                output_path = output_dir / img_name

                if resize_image(img_path, str(output_path), target_size):
                    processed_count += 1

            stats[f"{split_name}_{cls}"] = processed_count
            logger.info(
                f"Processed {processed_count} {cls} images for {split_name}"
            )

    # Log summary
    logger.info("Preprocessing completed. Summary:")
    for split in splits:
        cat_count = stats.get(f"{split}_cats", 0)
        dog_count = stats.get(f"{split}_dogs", 0)
        logger.info(f"  {split}: {cat_count} cats, {dog_count} dogs")

    return stats


if __name__ == "__main__":
    # Run preprocessing with default configuration
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.config import load_config, get_data_config

    config = load_config()
    data_config = get_data_config(config)

    preprocess_dataset(
        raw_dir=data_config.get("raw_dir", "data/raw"),
        processed_dir=data_config.get("processed_dir", "data/processed"),
        target_size=(
            data_config.get("image_size", 224),
            data_config.get("image_size", 224)
        ),
        train_ratio=data_config.get("train_split", 0.8),
        val_ratio=data_config.get("val_split", 0.1),
        test_ratio=data_config.get("test_split", 0.1),
        random_seed=data_config.get("random_seed", 42)
    )
