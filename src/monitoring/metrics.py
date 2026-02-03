"""
Metrics tracking module for the inference service.

This module provides functionality to track and expose metrics
such as request count, latency, and prediction statistics.
"""

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Thread-safe metrics collector for tracking API performance.

    Tracks the following metrics:
    - Total request count
    - Request count by endpoint
    - Request latency (min, max, avg)
    - Prediction distribution (cat vs dog)
    - Error count
    """

    def __init__(self):
        """Initialize the metrics collector with default values."""
        self._lock = threading.Lock()
        self._start_time = datetime.utcnow()

        # Request metrics
        self._total_requests = 0
        self._requests_by_endpoint = defaultdict(int)
        self._errors_by_endpoint = defaultdict(int)

        # Latency metrics (in milliseconds)
        self._latencies: List[float] = []
        self._max_latency_samples = 1000  # Keep last N samples

        # Prediction metrics
        self._predictions = defaultdict(int)
        self._total_predictions = 0

        # Confidence metrics
        self._confidence_sum = 0.0
        self._confidence_count = 0

    def record_request(self, endpoint: str, latency_ms: float, success: bool = True) -> None:
        """
        Record a request to the API.

        Args:
            endpoint: The endpoint that was called (e.g., "/predict", "/health")
            latency_ms: Request latency in milliseconds
            success: Whether the request was successful
        """
        with self._lock:
            self._total_requests += 1
            self._requests_by_endpoint[endpoint] += 1

            if not success:
                self._errors_by_endpoint[endpoint] += 1

            # Store latency sample
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_latency_samples:
                self._latencies.pop(0)

    def record_prediction(self, prediction: str, confidence: float) -> None:
        """
        Record a prediction result.

        Args:
            prediction: The predicted class ("cat" or "dog")
            confidence: Confidence score for the prediction
        """
        with self._lock:
            self._predictions[prediction] += 1
            self._total_predictions += 1
            self._confidence_sum += confidence
            self._confidence_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()

            # Calculate latency statistics
            latency_stats = self._calculate_latency_stats()

            # Calculate prediction distribution
            prediction_distribution = {}
            if self._total_predictions > 0:
                for pred, count in self._predictions.items():
                    prediction_distribution[pred] = {
                        "count": count,
                        "percentage": round(count / self._total_predictions * 100, 2),
                    }

            # Calculate average confidence
            avg_confidence = 0.0
            if self._confidence_count > 0:
                avg_confidence = self._confidence_sum / self._confidence_count

            # Calculate error rate
            total_errors = sum(self._errors_by_endpoint.values())
            error_rate = 0.0
            if self._total_requests > 0:
                error_rate = total_errors / self._total_requests * 100

            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "start_time": self._start_time.isoformat(),
                "requests": {
                    "total": self._total_requests,
                    "by_endpoint": dict(self._requests_by_endpoint),
                    "errors": dict(self._errors_by_endpoint),
                    "error_rate_percent": round(error_rate, 2),
                },
                "latency_ms": latency_stats,
                "predictions": {
                    "total": self._total_predictions,
                    "distribution": prediction_distribution,
                    "average_confidence": round(avg_confidence, 4),
                },
            }

    def _calculate_latency_stats(self) -> Dict[str, float]:
        """
        Calculate latency statistics from collected samples.

        Returns:
            Dictionary with min, max, avg, and p95 latency
        """
        if not self._latencies:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        return {
            "min": round(sorted_latencies[0], 2),
            "max": round(sorted_latencies[-1], 2),
            "avg": round(sum(sorted_latencies) / n, 2),
            "p50": round(sorted_latencies[int(n * 0.50)], 2),
            "p95": round(sorted_latencies[int(n * 0.95)], 2),
            "p99": round(sorted_latencies[int(n * 0.99)], 2),
        }

    def reset(self) -> None:
        """Reset all metrics to initial values."""
        with self._lock:
            self._start_time = datetime.utcnow()
            self._total_requests = 0
            self._requests_by_endpoint.clear()
            self._errors_by_endpoint.clear()
            self._latencies.clear()
            self._predictions.clear()
            self._total_predictions = 0
            self._confidence_sum = 0.0
            self._confidence_count = 0

        logger.info("Metrics have been reset")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics() -> Dict[str, Any]:
    """
    Get the current metrics from the global collector.

    Returns:
        Dictionary containing all collected metrics
    """
    return metrics_collector.get_metrics()


def record_request(endpoint: str, latency_ms: float, success: bool = True) -> None:
    """
    Record a request using the global metrics collector.

    Args:
        endpoint: The endpoint that was called
        latency_ms: Request latency in milliseconds
        success: Whether the request was successful
    """
    metrics_collector.record_request(endpoint, latency_ms, success)


def record_prediction(prediction: str, confidence: float) -> None:
    """
    Record a prediction using the global metrics collector.

    Args:
        prediction: The predicted class
        confidence: Confidence score for the prediction
    """
    metrics_collector.record_prediction(prediction, confidence)
