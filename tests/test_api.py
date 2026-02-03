"""
Unit tests for the FastAPI inference service.

These tests verify the API endpoints function correctly,
including health checks and prediction handling.
"""

import sys
import io
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Fixture to create a test client with mocked model loading
@pytest.fixture
def test_client():
    """Create a test client with mocked model."""
    # Set environment variable to prevent model loading
    os.environ["MODEL_PATH"] = "/nonexistent/model.pt"

    # Import and create test client
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    yield client


@pytest.fixture
def test_client_with_model():
    """Create a test client with a mock model loaded."""
    from src.models.cnn import SimpleCNN
    import src.api.app as app_module

    # Create a real model for testing
    mock_model = SimpleCNN(num_classes=2, dropout_rate=0.5)
    mock_model.eval()

    # Patch the global model variable
    original_model = app_module.model
    app_module.model = mock_model
    app_module.device = torch.device("cpu")

    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)

    yield client

    # Restore original
    app_module.model = original_model


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint_returns_200(self, test_client):
        """Test that health endpoint returns 200 status."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_correct_fields(self, test_client):
        """Test that health response contains required fields."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "version" in data

    def test_health_status_is_healthy(self, test_client):
        """Test that health status is 'healthy'."""
        response = test_client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"


class TestRootEndpoint:
    """Tests for the / (root) endpoint."""

    def test_root_endpoint_returns_200(self, test_client):
        """Test that root endpoint returns 200 status."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_contains_api_info(self, test_client):
        """Test that root response contains API information."""
        response = test_client.get("/")
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def create_test_image(self, format: str = "JPEG") -> bytes:
        """
        Create a test image in memory.

        Args:
            format: Image format (JPEG or PNG)

        Returns:
            Image bytes
        """
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()

    def test_predict_returns_503_when_model_not_loaded(self, test_client):
        """Test that predict returns 503 when model is not loaded."""
        image_bytes = self.create_test_image("JPEG")
        response = test_client.post(
            "/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        # Model not loaded should return 503
        assert response.status_code == 503

    def test_predict_rejects_non_image_file(self, test_client_with_model):
        """Test that predict rejects non-image files."""
        response = test_client_with_model.post(
            "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_rejects_empty_file(self, test_client_with_model):
        """Test that predict rejects empty files."""
        response = test_client_with_model.post(
            "/predict", files={"file": ("test.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400

    def test_predict_accepts_jpeg_with_model(self, test_client_with_model):
        """Test that predict accepts JPEG images when model is loaded."""
        image_bytes = self.create_test_image("JPEG")
        response = test_client_with_model.post(
            "/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_accepts_png_with_model(self, test_client_with_model):
        """Test that predict accepts PNG images when model is loaded."""
        image_bytes = self.create_test_image("PNG")
        response = test_client_with_model.post(
            "/predict", files={"file": ("test.png", image_bytes, "image/png")}
        )
        assert response.status_code == 200

    def test_predict_response_format(self, test_client_with_model):
        """Test that prediction response has correct format."""
        image_bytes = self.create_test_image("JPEG")
        response = test_client_with_model.post(
            "/predict", files={"file": ("test.jpg", image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "inference_time_ms" in data

        # Check prediction is valid class
        assert data["prediction"] in ["cat", "dog"]

        # Check confidence is valid probability
        assert 0 <= data["confidence"] <= 1

        # Check probabilities contain both classes
        assert "cat" in data["probabilities"]
        assert "dog" in data["probabilities"]


class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self, test_client):
        """Test that model info endpoint returns 200 status."""
        response = test_client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_shows_not_loaded_when_no_model(self, test_client):
        """Test that model info shows not loaded when model is absent."""
        response = test_client.get("/model/info")
        data = response.json()

        assert "loaded" in data
        assert data["loaded"] is False

    def test_model_info_shows_loaded_when_model_present(self, test_client_with_model):
        """Test that model info shows loaded when model is present."""
        response = test_client_with_model.get("/model/info")
        data = response.json()

        assert data["loaded"] is True
        assert "model_name" in data
        assert "num_classes" in data
        assert "total_parameters" in data


class TestPreprocessImage:
    """Tests for the preprocess_image function."""

    def test_preprocess_returns_tensor(self):
        """Test that preprocessing returns a PyTorch tensor."""
        from src.api.app import preprocess_image

        img = Image.new("RGB", (500, 300), color=(0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        tensor = preprocess_image(image_bytes)

        assert isinstance(tensor, torch.Tensor)

    def test_preprocess_returns_4d_tensor(self):
        """Test that preprocessing returns a 4D tensor (batch dimension)."""
        from src.api.app import preprocess_image

        img = Image.new("RGB", (500, 300), color=(0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        tensor = preprocess_image(image_bytes)

        assert len(tensor.shape) == 4
        assert tensor.shape[0] == 1  # Batch size
        assert tensor.shape[1] == 3  # Channels

    def test_preprocess_invalid_image_raises_error(self):
        """Test that invalid image data raises ValueError."""
        from src.api.app import preprocess_image

        with pytest.raises(ValueError):
            preprocess_image(b"not a valid image")


class TestLoadModelWeights:
    """Tests for the load_model_weights function."""

    def test_load_nonexistent_model_raises_error(self):
        """Test that loading nonexistent model raises FileNotFoundError."""
        from src.api.app import load_model_weights

        with pytest.raises(FileNotFoundError):
            load_model_weights("/nonexistent/path/model.pt")

    def test_load_model_returns_simplecnn(self, tmp_path):
        """Test that loading valid checkpoint returns SimpleCNN model."""
        from src.api.app import load_model_weights
        from src.models.cnn import SimpleCNN

        # Create a valid checkpoint
        model = SimpleCNN(num_classes=2, dropout_rate=0.5)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {"model": {"num_classes": 2, "dropout_rate": 0.5}},
        }

        model_path = tmp_path / "test_model.pt"
        torch.save(checkpoint, model_path)

        loaded_model = load_model_weights(str(model_path))

        assert isinstance(loaded_model, SimpleCNN)
        assert loaded_model.num_classes == 2

    def test_load_model_with_default_config(self, tmp_path):
        """Test loading model when config is not in checkpoint."""
        from src.api.app import load_model_weights
        from src.models.cnn import SimpleCNN

        # Create checkpoint without config
        model = SimpleCNN(num_classes=2, dropout_rate=0.5)
        checkpoint = {"model_state_dict": model.state_dict()}

        model_path = tmp_path / "test_model.pt"
        torch.save(checkpoint, model_path)

        loaded_model = load_model_weights(str(model_path))

        # Should use default values
        assert loaded_model.num_classes == 2
        assert loaded_model.dropout_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
