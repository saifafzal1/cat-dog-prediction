"""
Simple CNN model for Cats vs Dogs binary classification.

This module defines a baseline convolutional neural network
architecture for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for binary image classification.

    Architecture:
        - 3 Convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
        - Global Average Pooling
        - Fully connected classifier with dropout

    This is a baseline model suitable for the Cats vs Dogs classification task.
    Input images are expected to be 224x224 RGB images.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the SimpleCNN model.

        Args:
            num_classes: Number of output classes (default 2 for binary)
            dropout_rate: Dropout probability for regularization
        """
        super(SimpleCNN, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # First convolutional block
        # Input: 3 x 224 x 224, Output: 32 x 112 x 112
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        # Input: 32 x 112 x 112, Output: 64 x 56 x 56
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        # Input: 64 x 56 x 56, Output: 128 x 28 x 28
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        # Input: 128 x 28 x 28, Output: 256 x 14 x 14
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        # Input: 256 x 14 x 14, Output: 256
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Global average pooling and flatten
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions using softmax.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of probabilities).

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Prediction tensor of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def create_model(config: Dict[str, Any]) -> SimpleCNN:
    """
    Factory function to create a model from configuration.

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        Instantiated SimpleCNN model
    """
    model_config = config.get("model", {})

    model = SimpleCNN(
        num_classes=model_config.get("num_classes", 2),
        dropout_rate=model_config.get("dropout_rate", 0.5),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
