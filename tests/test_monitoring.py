"""
Unit tests for monitoring and metrics modules.

These tests verify the correctness of metrics collection,
logging, and performance tracking functionality.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.metrics import MetricsCollector
from src.monitoring.performance_tracker import PerformanceTracker, PredictionRecord


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_initialization(self):
        """Test that MetricsCollector initializes with correct defaults."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 0
        assert metrics["predictions"]["total"] == 0
        assert "uptime_seconds" in metrics

    def test_record_request(self):
        """Test recording a single request."""
        collector = MetricsCollector()

        collector.record_request("/health", 50.0, success=True)
        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 1
        assert metrics["requests"]["by_endpoint"]["/health"] == 1
        assert metrics["latency_ms"]["avg"] == 50.0

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        collector = MetricsCollector()

        collector.record_request("/health", 50.0)
        collector.record_request("/predict", 100.0)
        collector.record_request("/predict", 150.0)

        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 3
        assert metrics["requests"]["by_endpoint"]["/health"] == 1
        assert metrics["requests"]["by_endpoint"]["/predict"] == 2

    def test_record_failed_request(self):
        """Test recording a failed request."""
        collector = MetricsCollector()

        collector.record_request("/predict", 100.0, success=False)
        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 1
        assert metrics["requests"]["errors"]["/predict"] == 1
        assert metrics["requests"]["error_rate_percent"] == 100.0

    def test_record_prediction(self):
        """Test recording predictions."""
        collector = MetricsCollector()

        collector.record_prediction("cat", 0.95)
        collector.record_prediction("dog", 0.85)
        collector.record_prediction("cat", 0.90)

        metrics = collector.get_metrics()

        assert metrics["predictions"]["total"] == 3
        assert metrics["predictions"]["distribution"]["cat"]["count"] == 2
        assert metrics["predictions"]["distribution"]["dog"]["count"] == 1

    def test_latency_statistics(self):
        """Test latency statistics calculation."""
        collector = MetricsCollector()

        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        for lat in latencies:
            collector.record_request("/test", lat)

        metrics = collector.get_metrics()

        assert metrics["latency_ms"]["min"] == 10.0
        assert metrics["latency_ms"]["max"] == 50.0
        assert metrics["latency_ms"]["avg"] == 30.0

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        collector.record_request("/test", 100.0)
        collector.record_prediction("cat", 0.9)

        collector.reset()
        metrics = collector.get_metrics()

        assert metrics["requests"]["total"] == 0
        assert metrics["predictions"]["total"] == 0

    def test_thread_safety(self):
        """Test that metrics collection is thread-safe."""
        import threading

        collector = MetricsCollector()
        num_threads = 10
        requests_per_thread = 100

        def record_requests():
            for _ in range(requests_per_thread):
                collector.record_request("/test", 10.0)

        threads = [threading.Thread(target=record_requests) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = collector.get_metrics()
        assert metrics["requests"]["total"] == num_threads * requests_per_thread


class TestPerformanceTracker:
    """Tests for the PerformanceTracker class."""

    def test_initialization(self):
        """Test that PerformanceTracker initializes correctly."""
        tracker = PerformanceTracker(max_records=100)
        metrics = tracker.get_performance_metrics()

        assert metrics["total_predictions"] == 0
        assert metrics["labeled_predictions"] == 0

    def test_record_prediction(self):
        """Test recording a prediction."""
        tracker = PerformanceTracker(max_records=100)

        tracker.record_prediction(
            prediction="cat",
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05}
        )

        metrics = tracker.get_performance_metrics()

        assert metrics["total_predictions"] == 1
        assert "cat" in metrics["prediction_distribution"]

    def test_record_prediction_with_label(self):
        """Test recording a prediction with true label."""
        tracker = PerformanceTracker(max_records=100)

        tracker.record_prediction(
            prediction="cat",
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05},
            true_label="cat"
        )

        metrics = tracker.get_performance_metrics()

        assert metrics["labeled_predictions"] == 1
        assert metrics["accuracy_metrics"]["accuracy"] == 1.0

    def test_accuracy_calculation(self):
        """Test accuracy calculation with mixed predictions."""
        tracker = PerformanceTracker(max_records=100)

        # 3 correct, 2 incorrect
        tracker.record_prediction("cat", 0.9, {"cat": 0.9, "dog": 0.1}, true_label="cat")
        tracker.record_prediction("dog", 0.85, {"cat": 0.15, "dog": 0.85}, true_label="dog")
        tracker.record_prediction("cat", 0.8, {"cat": 0.8, "dog": 0.2}, true_label="cat")
        tracker.record_prediction("cat", 0.7, {"cat": 0.7, "dog": 0.3}, true_label="dog")  # Wrong
        tracker.record_prediction("dog", 0.75, {"cat": 0.25, "dog": 0.75}, true_label="cat")  # Wrong

        metrics = tracker.get_performance_metrics()

        assert metrics["labeled_predictions"] == 5
        assert metrics["accuracy_metrics"]["accuracy"] == 0.6  # 3/5

    def test_add_true_label(self):
        """Test adding a true label to an existing prediction."""
        tracker = PerformanceTracker(max_records=100)

        tracker.record_prediction("cat", 0.9, {"cat": 0.9, "dog": 0.1})
        tracker.add_true_label(-1, "cat")

        metrics = tracker.get_performance_metrics()

        assert metrics["labeled_predictions"] == 1
        assert metrics["accuracy_metrics"]["accuracy"] == 1.0

    def test_get_recent_predictions(self):
        """Test getting recent predictions."""
        tracker = PerformanceTracker(max_records=100)

        for i in range(5):
            tracker.record_prediction(
                prediction="cat" if i % 2 == 0 else "dog",
                confidence=0.9,
                probabilities={"cat": 0.5, "dog": 0.5}
            )

        recent = tracker.get_recent_predictions(3)

        assert len(recent) == 3

    def test_confidence_statistics(self):
        """Test confidence statistics calculation."""
        tracker = PerformanceTracker(max_records=100)

        tracker.record_prediction("cat", 0.9, {"cat": 0.9, "dog": 0.1})
        tracker.record_prediction("cat", 0.8, {"cat": 0.8, "dog": 0.2})
        tracker.record_prediction("cat", 0.7, {"cat": 0.7, "dog": 0.3})

        metrics = tracker.get_performance_metrics()

        assert metrics["confidence_statistics"]["min"] == 0.7
        assert metrics["confidence_statistics"]["max"] == 0.9
        assert abs(metrics["confidence_statistics"]["mean"] - 0.8) < 0.01

    def test_max_records_limit(self):
        """Test that records are limited to max_records."""
        tracker = PerformanceTracker(max_records=5)

        for i in range(10):
            tracker.record_prediction("cat", 0.9, {"cat": 0.9, "dog": 0.1})

        metrics = tracker.get_performance_metrics()

        # Should only keep last 5 records
        assert metrics["total_predictions"] == 5

    def test_clear(self):
        """Test clearing the tracker."""
        tracker = PerformanceTracker(max_records=100)

        tracker.record_prediction("cat", 0.9, {"cat": 0.9, "dog": 0.1}, true_label="cat")
        tracker.clear()

        metrics = tracker.get_performance_metrics()

        assert metrics["total_predictions"] == 0
        assert metrics["labeled_predictions"] == 0


class TestPredictionRecord:
    """Tests for the PredictionRecord class."""

    def test_to_dict(self):
        """Test converting prediction record to dictionary."""
        now = datetime.utcnow()
        record = PredictionRecord(
            timestamp=now,
            prediction="cat",
            confidence=0.95,
            probabilities={"cat": 0.95, "dog": 0.05},
            true_label="cat"
        )

        data = record.to_dict()

        assert data["prediction"] == "cat"
        assert data["confidence"] == 0.95
        assert data["true_label"] == "cat"
        assert "timestamp" in data

    def test_record_without_label(self):
        """Test prediction record without true label."""
        record = PredictionRecord(
            timestamp=datetime.utcnow(),
            prediction="dog",
            confidence=0.8,
            probabilities={"cat": 0.2, "dog": 0.8}
        )

        data = record.to_dict()

        assert data["true_label"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
