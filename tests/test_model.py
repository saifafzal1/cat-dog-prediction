"""
Unit tests for model utility and inference functions.

These tests verify the correctness of the CNN model architecture,
forward pass, and utility functions.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.cnn import SimpleCNN, count_parameters, create_model  # noqa: E402


class TestSimpleCNN:
    """Tests for the SimpleCNN model class."""

    def test_model_initialization(self):
        """Test that the model initializes correctly with default parameters."""
        model = SimpleCNN()

        assert model.num_classes == 2
        assert model.dropout_rate == 0.5
        assert isinstance(model, nn.Module)

    def test_model_custom_parameters(self):
        """Test model initialization with custom parameters."""
        model = SimpleCNN(num_classes=10, dropout_rate=0.3)

        assert model.num_classes == 10
        assert model.dropout_rate == 0.3

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = SimpleCNN(num_classes=2)
        model.eval()

        # Create dummy input (batch_size=4, channels=3, height=224, width=224)
        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 2)

    def test_forward_pass_single_image(self):
        """Test forward pass with a single image."""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2)

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SimpleCNN()
        model.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for input
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_predict_proba(self):
        """Test that predict_proba returns valid probabilities."""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            probs = model.predict_proba(x)

        # Check shape
        assert probs.shape == (4, 2)

        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-6)

        # Check probabilities are in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict(self):
        """Test that predict returns valid class indices."""
        model = SimpleCNN()
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            predictions = model.predict(x)

        # Check shape
        assert predictions.shape == (4,)

        # Check predictions are valid class indices
        assert (predictions >= 0).all()
        assert (predictions < 2).all()

    def test_model_layers_exist(self):
        """Test that all expected layers are present in the model."""
        model = SimpleCNN()

        # Check convolutional layers
        assert hasattr(model, "conv1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "conv4")

        # Check batch normalization layers
        assert hasattr(model, "bn1")
        assert hasattr(model, "bn2")
        assert hasattr(model, "bn3")
        assert hasattr(model, "bn4")

        # Check pooling layers
        assert hasattr(model, "pool1")
        assert hasattr(model, "pool2")
        assert hasattr(model, "pool3")
        assert hasattr(model, "pool4")

        # Check fully connected layers
        assert hasattr(model, "fc1")
        assert hasattr(model, "fc2")
        assert hasattr(model, "dropout")

    def test_model_device_transfer(self):
        """Test that model can be transferred to different devices."""
        model = SimpleCNN()

        # Test CPU
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # GPU test would require CUDA availability
        if torch.cuda.is_available():
            model_gpu = model.to("cuda")
            assert next(model_gpu.parameters()).device.type == "cuda"


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_model_default_config(self):
        """Test model creation with empty config."""
        config = {}
        model = create_model(config)

        assert isinstance(model, SimpleCNN)
        assert model.num_classes == 2
        assert model.dropout_rate == 0.5

    def test_create_model_custom_config(self):
        """Test model creation with custom config."""
        config = {"model": {"num_classes": 5, "dropout_rate": 0.3}}
        model = create_model(config)

        assert model.num_classes == 5
        assert model.dropout_rate == 0.3

    def test_create_model_partial_config(self):
        """Test model creation with partial config."""
        config = {
            "model": {
                "num_classes": 3
                # dropout_rate not specified, should use default
            }
        }
        model = create_model(config)

        assert model.num_classes == 3
        assert model.dropout_rate == 0.5  # Default value


class TestCountParameters:
    """Tests for the count_parameters utility function."""

    def test_count_parameters_returns_int(self):
        """Test that count_parameters returns an integer."""
        model = SimpleCNN()
        param_count = count_parameters(model)

        assert isinstance(param_count, int)

    def test_count_parameters_positive(self):
        """Test that parameter count is positive."""
        model = SimpleCNN()
        param_count = count_parameters(model)

        assert param_count > 0

    def test_count_parameters_consistency(self):
        """Test that count is consistent for same model architecture."""
        model1 = SimpleCNN()
        model2 = SimpleCNN()

        count1 = count_parameters(model1)
        count2 = count_parameters(model2)

        assert count1 == count2

    def test_count_parameters_different_configs(self):
        """Test that different configs produce different parameter counts."""
        model_small = SimpleCNN(num_classes=2)
        model_large = SimpleCNN(num_classes=100)

        count_small = count_parameters(model_small)
        count_large = count_parameters(model_large)

        # More classes means more parameters in the final layer
        assert count_large > count_small

    def test_count_parameters_frozen_layers(self):
        """Test that frozen parameters are not counted."""
        model = SimpleCNN()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        frozen_count = count_parameters(model)
        assert frozen_count == 0

        # Unfreeze and verify count is positive
        for param in model.parameters():
            param.requires_grad = True

        unfrozen_count = count_parameters(model)
        assert unfrozen_count > 0


class TestModelInference:
    """Tests for model inference behavior."""

    def test_eval_mode_disables_dropout(self):
        """Test that eval mode disables dropout (deterministic output)."""
        model = SimpleCNN(dropout_rate=0.5)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # In eval mode, outputs should be identical
        assert torch.allclose(output1, output2)

    def test_train_mode_enables_dropout(self):
        """Test that train mode enables dropout (stochastic output)."""
        model = SimpleCNN(dropout_rate=0.5)
        model.train()

        x = torch.randn(2, 3, 224, 224)

        # Run multiple times and check for variation
        outputs = []
        for _ in range(10):
            output = model(x)
            outputs.append(output.detach().clone())

        # At least some outputs should be different due to dropout
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same

    def test_batch_independence(self):
        """Test that predictions for each sample are independent."""
        model = SimpleCNN()
        model.eval()

        # Create two different images
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            # Process individually
            out1_single = model(x1)
            out2_single = model(x2)

            # Process as batch
            batch = torch.cat([x1, x2], dim=0)
            out_batch = model(batch)

        # Individual and batched results should match
        assert torch.allclose(out1_single, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out2_single, out_batch[1:2], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
