"""
Unit tests for data preprocessing functions.

These tests verify the correctness of image validation, resizing,
and dataset splitting functions.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocess import resize_image, split_dataset, validate_image  # noqa: E402


class TestValidateImage:
    """Tests for the validate_image function."""

    def test_validate_valid_jpeg(self, tmp_path):
        """Test that a valid JPEG image passes validation."""
        # Create a valid test image
        img_path = tmp_path / "test_image.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path, "JPEG")

        assert validate_image(str(img_path)) is True

    def test_validate_valid_png(self, tmp_path):
        """Test that a valid PNG image passes validation."""
        img_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(img_path, "PNG")

        assert validate_image(str(img_path)) is True

    def test_validate_nonexistent_file(self):
        """Test that a nonexistent file fails validation."""
        assert validate_image("/nonexistent/path/image.jpg") is False

    def test_validate_corrupted_file(self, tmp_path):
        """Test that a corrupted file fails validation."""
        # Create a corrupted file (not a valid image)
        corrupted_path = tmp_path / "corrupted.jpg"
        with open(corrupted_path, "wb") as f:
            f.write(b"This is not a valid image file")

        assert validate_image(str(corrupted_path)) is False

    def test_validate_grayscale_image(self, tmp_path):
        """Test that a grayscale image passes validation."""
        img_path = tmp_path / "grayscale.jpg"
        img = Image.new("L", (100, 100), color=128)
        img.save(img_path, "JPEG")

        assert validate_image(str(img_path)) is True


class TestResizeImage:
    """Tests for the resize_image function."""

    def test_resize_to_target_size(self, tmp_path):
        """Test that image is resized to the correct target size."""
        # Create source image
        src_path = tmp_path / "source.jpg"
        img = Image.new("RGB", (500, 300), color="green")
        img.save(src_path, "JPEG")

        # Resize
        dst_path = tmp_path / "resized.jpg"
        target_size = (224, 224)
        result = resize_image(str(src_path), str(dst_path), target_size)

        assert result is True
        assert dst_path.exists()

        # Verify dimensions
        resized_img = Image.open(dst_path)
        assert resized_img.size == target_size

    def test_resize_converts_grayscale_to_rgb(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        # Create grayscale source image
        src_path = tmp_path / "grayscale.jpg"
        img = Image.new("L", (100, 100), color=128)
        img.save(src_path, "JPEG")

        # Resize
        dst_path = tmp_path / "resized.jpg"
        result = resize_image(str(src_path), str(dst_path), (224, 224))

        assert result is True

        # Verify mode is RGB
        resized_img = Image.open(dst_path)
        assert resized_img.mode == "RGB"

    def test_resize_converts_rgba_to_rgb(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        # Create RGBA source image
        src_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        img.save(src_path, "PNG")

        # Resize
        dst_path = tmp_path / "resized.jpg"
        result = resize_image(str(src_path), str(dst_path), (224, 224))

        assert result is True

        resized_img = Image.open(dst_path)
        assert resized_img.mode == "RGB"

    def test_resize_nonexistent_file(self, tmp_path):
        """Test that resizing a nonexistent file returns False."""
        dst_path = tmp_path / "output.jpg"
        result = resize_image("/nonexistent/image.jpg", str(dst_path), (224, 224))

        assert result is False

    def test_resize_preserves_aspect_ratio_fill(self, tmp_path):
        """Test that resize fills the target dimensions."""
        src_path = tmp_path / "source.jpg"
        img = Image.new("RGB", (640, 480), color="yellow")
        img.save(src_path, "JPEG")

        dst_path = tmp_path / "resized.jpg"
        target_size = (224, 224)
        resize_image(str(src_path), str(dst_path), target_size)

        resized_img = Image.open(dst_path)
        # The image should be exactly the target size (stretched/scaled)
        assert resized_img.size == target_size


class TestSplitDataset:
    """Tests for the split_dataset function."""

    def test_split_ratios(self):
        """Test that dataset is split according to specified ratios."""
        # Create sample paths
        paths = [f"image_{i}.jpg" for i in range(100)]

        train, val, test = split_dataset(
            paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42
        )

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_split_no_overlap(self):
        """Test that there is no overlap between splits."""
        paths = [f"image_{i}.jpg" for i in range(100)]

        train, val, test = split_dataset(paths, random_seed=42)

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        # Check no overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_all_elements_included(self):
        """Test that all elements are included in some split."""
        paths = [f"image_{i}.jpg" for i in range(100)]

        train, val, test = split_dataset(paths, random_seed=42)

        all_split = set(train) | set(val) | set(test)
        assert all_split == set(paths)

    def test_split_reproducibility(self):
        """Test that the same seed produces the same split."""
        paths = [f"image_{i}.jpg" for i in range(100)]

        train1, val1, test1 = split_dataset(paths, random_seed=42)
        train2, val2, test2 = split_dataset(paths, random_seed=42)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_split_different_seeds_different_results(self):
        """Test that different seeds produce different splits."""
        paths = [f"image_{i}.jpg" for i in range(100)]

        train1, _, _ = split_dataset(paths, random_seed=42)
        train2, _, _ = split_dataset(paths, random_seed=123)

        # Very unlikely to be the same with different seeds
        assert train1 != train2

    def test_split_invalid_ratios(self):
        """Test that invalid ratios raise ValueError."""
        paths = [f"image_{i}.jpg" for i in range(100)]

        with pytest.raises(ValueError):
            split_dataset(paths, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)  # Sum > 1.0

    def test_split_empty_list(self):
        """Test splitting an empty list."""
        train, val, test = split_dataset([], random_seed=42)

        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0

    def test_split_small_dataset(self):
        """Test splitting a very small dataset."""
        paths = ["image_0.jpg", "image_1.jpg", "image_2.jpg"]

        train, val, test = split_dataset(
            paths, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42
        )

        # All paths should be distributed
        total = len(train) + len(val) + len(test)
        assert total == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
