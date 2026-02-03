"""
Model performance tracking module for post-deployment monitoring.

This module tracks model predictions and allows comparison with
true labels to detect model drift and performance degradation.
"""

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import logging

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class PredictionRecord:
    """
    Data class for storing a single prediction record.
    """

    def __init__(
        self,
        timestamp: datetime,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        true_label: Optional[str] = None,
    ):
        """
        Initialize a prediction record.

        Args:
            timestamp: When the prediction was made
            prediction: Predicted class
            confidence: Confidence score
            probabilities: Class probabilities
            true_label: Ground truth label (if available)
        """
        self.timestamp = timestamp
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.true_label = true_label

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "true_label": self.true_label,
        }


class PerformanceTracker:
    """
    Tracks model performance metrics over time.

    Stores predictions and true labels to calculate:
    - Accuracy (when labels are provided)
    - Confidence distribution
    - Prediction distribution drift
    """

    def __init__(self, max_records: int = 10000, storage_path: Optional[str] = None):
        """
        Initialize the performance tracker.

        Args:
            max_records: Maximum number of records to keep in memory
            storage_path: Optional path to persist records
        """
        self._lock = threading.Lock()
        self._records: deque = deque(maxlen=max_records)
        self._storage_path = storage_path
        self._labeled_count = 0
        self._correct_count = 0

        # Load existing records if storage path exists
        if storage_path and Path(storage_path).exists():
            self._load_records()

    def record_prediction(
        self,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        true_label: Optional[str] = None,
    ) -> None:
        """
        Record a new prediction.

        Args:
            prediction: Predicted class
            confidence: Confidence score
            probabilities: Class probabilities
            true_label: Ground truth label (if available)
        """
        record = PredictionRecord(
            timestamp=datetime.utcnow(),
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            true_label=true_label,
        )

        with self._lock:
            self._records.append(record)

            # Update accuracy tracking if label is provided
            if true_label is not None:
                self._labeled_count += 1
                if prediction == true_label:
                    self._correct_count += 1

        logger.debug(f"Recorded prediction: {prediction} (label: {true_label})")

    def add_true_label(self, index: int, true_label: str) -> bool:
        """
        Add a true label to an existing prediction record.

        Args:
            index: Index of the record (negative indexing supported)
            true_label: Ground truth label

        Returns:
            True if label was added successfully
        """
        with self._lock:
            try:
                record = self._records[index]
                old_label = record.true_label

                record.true_label = true_label

                # Update accuracy tracking
                if old_label is None:
                    self._labeled_count += 1
                    if record.prediction == true_label:
                        self._correct_count += 1
                elif old_label != true_label:
                    # Label was changed, update correct count
                    if old_label == record.prediction:
                        self._correct_count -= 1
                    if true_label == record.prediction:
                        self._correct_count += 1

                return True
            except IndexError:
                logger.error(f"Invalid record index: {index}")
                return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return current performance metrics.

        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            if not self._records:
                return {
                    "total_predictions": 0,
                    "labeled_predictions": 0,
                    "accuracy": None,
                    "message": "No predictions recorded yet",
                }

            records = list(self._records)

        # Calculate metrics
        total = len(records)
        predictions = [r.prediction for r in records]
        confidences = [r.confidence for r in records]
        labeled = [r for r in records if r.true_label is not None]

        # Prediction distribution
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        prediction_distribution = {
            pred: {"count": count, "percentage": round(count / total * 100, 2)}
            for pred, count in pred_counts.items()
        }

        # Confidence statistics
        confidence_stats = {
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "mean": round(np.mean(confidences), 4),
            "std": round(np.std(confidences), 4),
        }

        # Accuracy metrics (only if we have labeled data)
        accuracy_metrics = None
        if labeled:
            correct = sum(1 for r in labeled if r.prediction == r.true_label)
            accuracy_metrics = {
                "labeled_count": len(labeled),
                "correct_count": correct,
                "accuracy": round(correct / len(labeled), 4),
                "error_rate": round(1 - (correct / len(labeled)), 4),
            }

            # Per-class metrics
            class_metrics = self._calculate_class_metrics(labeled)
            accuracy_metrics["per_class"] = class_metrics

        # Time range
        timestamps = [r.timestamp for r in records]
        time_range = {
            "first_prediction": min(timestamps).isoformat(),
            "last_prediction": max(timestamps).isoformat(),
        }

        return {
            "total_predictions": total,
            "labeled_predictions": len(labeled),
            "prediction_distribution": prediction_distribution,
            "confidence_statistics": confidence_stats,
            "accuracy_metrics": accuracy_metrics,
            "time_range": time_range,
        }

    def _calculate_class_metrics(
        self, labeled_records: List[PredictionRecord]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class precision, recall, and F1 score.

        Args:
            labeled_records: List of records with true labels

        Returns:
            Dictionary with metrics for each class
        """
        classes = set()
        for r in labeled_records:
            classes.add(r.prediction)
            classes.add(r.true_label)

        metrics = {}
        for cls in classes:
            # True positives, false positives, false negatives
            tp = sum(1 for r in labeled_records if r.prediction == cls and r.true_label == cls)
            fp = sum(1 for r in labeled_records if r.prediction == cls and r.true_label != cls)
            fn = sum(1 for r in labeled_records if r.prediction != cls and r.true_label == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[cls] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "support": tp + fn,
            }

        return metrics

    def get_recent_predictions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most recent predictions.

        Args:
            n: Number of predictions to return

        Returns:
            List of prediction records as dictionaries
        """
        with self._lock:
            recent = list(self._records)[-n:]

        return [r.to_dict() for r in recent]

    def save_records(self, filepath: Optional[str] = None) -> None:
        """
        Save prediction records to a JSON file.

        Args:
            filepath: Path to save file (uses default if not specified)
        """
        save_path = filepath or self._storage_path
        if not save_path:
            logger.warning("No storage path specified, cannot save records")
            return

        with self._lock:
            records_data = [r.to_dict() for r in self._records]

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(records_data, f, indent=2)

        logger.info(f"Saved {len(records_data)} records to {save_path}")

    def _load_records(self) -> None:
        """Load prediction records from storage."""
        if not self._storage_path or not Path(self._storage_path).exists():
            return

        try:
            with open(self._storage_path, "r") as f:
                records_data = json.load(f)

            for data in records_data:
                record = PredictionRecord(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    prediction=data["prediction"],
                    confidence=data["confidence"],
                    probabilities=data["probabilities"],
                    true_label=data.get("true_label"),
                )
                self._records.append(record)

                if record.true_label is not None:
                    self._labeled_count += 1
                    if record.prediction == record.true_label:
                        self._correct_count += 1

            logger.info(f"Loaded {len(self._records)} records from {self._storage_path}")
        except Exception as e:
            logger.error(f"Failed to load records: {e}")

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self._records.clear()
            self._labeled_count = 0
            self._correct_count = 0

        logger.info("Performance tracker cleared")


# Global performance tracker instance
performance_tracker = PerformanceTracker(
    max_records=10000, storage_path=os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.json")
)
