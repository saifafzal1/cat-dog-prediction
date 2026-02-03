"""
Training script for Cats vs Dogs classification model.

This script handles model training with MLflow experiment tracking,
including logging of parameters, metrics, and artifacts.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import mlflow
import mlflow.pytorch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_training_config, get_mlflow_config
from src.data.dataloader import create_data_loaders, get_class_names
from src.models.cnn import create_model, count_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        device: Device to run training on (CPU/GPU)
        epoch: Current epoch number (for logging)

    Returns:
        Tuple of (average_loss, accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for training
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc": f"{100 * correct / total:.2f}%",
            }
        )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    phase: str = "Val",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        phase: Name of the evaluation phase for logging

    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"[{phase}]")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions for metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / len(data_loader):.4f}",
                    "acc": f"{100 * correct / total:.2f}%",
                }
            )

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: str
) -> None:
    """
    Plot and save a confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_training_curves(
    train_losses: list, val_losses: list, train_accs: list, val_accs: list, save_path: str
) -> None:
    """
    Plot and save training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Training curves saved to {save_path}")


def train(config: Dict[str, Any], data_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Main training function with MLflow tracking.

    This function orchestrates the complete training process including:
    - Model creation
    - Data loading
    - Training loop with validation
    - MLflow experiment tracking
    - Model and artifact saving

    Args:
        config: Full configuration dictionary
        data_dir: Directory containing processed train/val/test data
        output_dir: Directory to save trained models and artifacts

    Returns:
        Dictionary containing training results and metrics
    """
    # Extract configuration
    training_config = get_training_config(config)
    mlflow_config = get_mlflow_config(config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_config.get("experiment_name", "cats_vs_dogs"))

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    logger.info("Creating data loaders...")
    loaders = create_data_loaders(data_dir, config)

    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
    )

    # Learning rate scheduler
    scheduler_config = training_config.get("scheduler", {})
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_config.get("step_size", 5),
        gamma=scheduler_config.get("gamma", 0.1),
    )

    # Training parameters
    epochs = training_config.get("epochs", 10)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "model_name": config.get("model", {}).get("name", "SimpleCNN"),
                "num_classes": config.get("model", {}).get("num_classes", 2),
                "dropout_rate": config.get("model", {}).get("dropout_rate", 0.5),
                "batch_size": training_config.get("batch_size", 32),
                "epochs": epochs,
                "learning_rate": training_config.get("learning_rate", 0.001),
                "weight_decay": training_config.get("weight_decay", 0.0001),
                "optimizer": training_config.get("optimizer", "adam"),
                "device": str(device),
                "total_parameters": count_parameters(model),
            }
        )

        # Training history
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0.0

        # Training loop
        logger.info("Starting training...")
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], criterion, optimizer, device, epoch
            )

            # Validate
            val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device, "Val")

            # Update scheduler
            scheduler.step()

            # Record history
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = output_path / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "config": config,
                    },
                    best_model_path,
                )
                logger.info(f"Best model saved with val_acc: {val_acc:.2f}%")

        # Final evaluation on test set
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_labels, test_preds = evaluate(
            model, loaders["test"], criterion, device, "Test"
        )

        # Calculate detailed metrics
        class_names = list(get_class_names().values())
        precision = precision_score(test_labels, test_preds, average="binary")
        recall = recall_score(test_labels, test_preds, average="binary")
        f1 = f1_score(test_labels, test_preds, average="binary")

        # Log final test metrics
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

        logger.info(
            f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

        # Save and log artifacts
        # Confusion matrix
        cm_path = output_path / "confusion_matrix.png"
        plot_confusion_matrix(test_labels, test_preds, class_names, str(cm_path))
        mlflow.log_artifact(str(cm_path))

        # Training curves
        curves_path = output_path / "training_curves.png"
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, str(curves_path))
        mlflow.log_artifact(str(curves_path))

        # Save final model
        final_model_path = output_path / "final_model.pt"
        torch.save(
            {
                "epoch": epochs - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accs[-1],
                "test_accuracy": test_acc,
                "config": config,
            },
            final_model_path,
        )

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        # Save classification report
        report = classification_report(
            test_labels, test_preds, target_names=class_names, output_dict=True
        )
        report_path = output_path / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        logger.info(f"Training completed. Artifacts saved to {output_path}")

        # Return results
        results = {
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "model_path": str(final_model_path),
            "best_model_path": str(output_path / "best_model.pt"),
        }

        return results


if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Get data and output directories from config
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    data_dir = data_config.get("processed_dir", "data/processed")
    output_dir = output_config.get("model_dir", "models")

    # Run training
    results = train(config, data_dir, output_dir)

    # Print final results
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Test F1 Score: {results['test_f1']:.4f}")
    print(f"Model saved to: {results['model_path']}")
