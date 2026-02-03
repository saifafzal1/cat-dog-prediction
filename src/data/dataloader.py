"""
PyTorch DataLoader module for Cats vs Dogs classification.

This module provides dataset classes and data loaders with
appropriate augmentation for training and evaluation.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CatsDogsDataset(Dataset):
    """
    PyTorch Dataset for Cats vs Dogs classification.

    This dataset loads images from a directory structure where
    each class has its own subdirectory.

    Attributes:
        root_dir: Root directory containing class subdirectories
        transform: Torchvision transforms to apply to images
        class_to_idx: Mapping from class name to numeric index
        samples: List of (image_path, label) tuples
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Path to the directory containing class subdirectories
                     (e.g., data/processed/train)
            transform: Optional transforms to apply to each image
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Define class mapping (alphabetical order for consistency)
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all samples
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        """
        Load all image paths and their corresponding labels.

        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        image_extensions = {".jpg", ".jpeg", ".png"}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    samples.append((str(img_path), class_idx))

        return samples

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load and convert image to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Create training data transforms with augmentation.

    Training transforms include random augmentation for better
    model generalization.

    Args:
        config: Configuration dictionary containing augmentation parameters

    Returns:
        Composed transforms for training data
    """
    aug_config = config.get("augmentation", {})
    normalize_config = aug_config.get("normalize", {})

    # Build list of transforms
    transform_list = []

    # Random horizontal flip
    if aug_config.get("horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Random rotation
    rotation_degrees = aug_config.get("rotation_degrees", 15)
    if rotation_degrees > 0:
        transform_list.append(
            transforms.RandomRotation(degrees=rotation_degrees)
        )

    # Color jitter for brightness, contrast, saturation, and hue
    color_config = aug_config.get("color_jitter", {})
    if color_config:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_config.get("brightness", 0.2),
                contrast=color_config.get("contrast", 0.2),
                saturation=color_config.get("saturation", 0.2),
                hue=color_config.get("hue", 0.1)
            )
        )

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalize with ImageNet statistics
    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


def get_eval_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Create evaluation data transforms (validation and test).

    Evaluation transforms only include normalization without augmentation
    to ensure consistent and reproducible results.

    Args:
        config: Configuration dictionary containing normalization parameters

    Returns:
        Composed transforms for evaluation data
    """
    aug_config = config.get("augmentation", {})
    normalize_config = aug_config.get("normalize", {})

    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def create_data_loaders(
    data_dir: str,
    config: Dict[str, Any],
    batch_size: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.

    Args:
        data_dir: Base directory containing train/val/test subdirectories
        config: Full configuration dictionary
        batch_size: Override batch size from config if specified

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoader objects
    """
    data_path = Path(data_dir)
    training_config = config.get("training", {})

    # Get batch size
    if batch_size is None:
        batch_size = training_config.get("batch_size", 32)

    # Create transforms
    train_transforms = get_train_transforms(config)
    eval_transforms = get_eval_transforms(config)

    # Create datasets
    train_dataset = CatsDogsDataset(
        root_dir=data_path / "train",
        transform=train_transforms
    )
    val_dataset = CatsDogsDataset(
        root_dir=data_path / "val",
        transform=eval_transforms
    )
    test_dataset = CatsDogsDataset(
        root_dir=data_path / "test",
        transform=eval_transforms
    )

    # Create data loaders
    # Use num_workers=0 for compatibility, increase for faster loading
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    }

    return loaders


def get_class_names() -> Dict[int, str]:
    """
    Get the mapping from class index to class name.

    Returns:
        Dictionary mapping index to class name
    """
    return {0: "cat", 1: "dog"}
