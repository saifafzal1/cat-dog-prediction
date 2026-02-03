"""
Logging configuration module for the inference service.

This module provides structured logging configuration with support
for different log levels, formats, and output destinations.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing
    by log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data)


class RequestLogFilter(logging.Filter):
    """
    Filter to add request context to log records.

    Adds request-specific information like request ID
    to all log records.
    """

    def __init__(self, name: str = ""):
        """Initialize the filter."""
        super().__init__(name)
        self.request_id: Optional[str] = None

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request ID to the log record.

        Args:
            record: The log record

        Returns:
            True (always passes the record)
        """
        record.request_id = self.request_id or "N/A"
        return True


def setup_logging(
    log_level: str = "INFO", log_format: str = "standard", log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("standard" or "json")
        log_file: Optional path to log file
    """
    # Get log level from environment or parameter
    level = os.environ.get("LOG_LEVEL", log_level).upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class RequestLogger:
    """
    Request logger for tracking API requests and responses.

    Logs request details while excluding sensitive information.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the request logger.

        Args:
            logger: Optional logger instance to use
        """
        self.logger = logger or logging.getLogger("request_logger")

    def log_request(
        self,
        method: str,
        path: str,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an incoming request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            client_ip: Client IP address (anonymized)
            request_id: Unique request identifier
            extra: Additional data to log
        """
        log_data = {
            "event": "request_received",
            "method": method,
            "path": path,
            "request_id": request_id,
        }

        # Anonymize IP address (keep first two octets only)
        if client_ip:
            parts = client_ip.split(".")
            if len(parts) == 4:
                log_data["client_ip"] = f"{parts[0]}.{parts[1]}.x.x"

        if extra:
            log_data.update(extra)

        self.logger.info(f"Request: {method} {path}", extra={"extra_data": log_data})

    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        request_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a response.

        Args:
            method: HTTP method
            path: Request path
            status_code: HTTP status code
            latency_ms: Request latency in milliseconds
            request_id: Unique request identifier
            extra: Additional data to log
        """
        log_data = {
            "event": "response_sent",
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "request_id": request_id,
        }

        if extra:
            log_data.update(extra)

        log_level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            log_level,
            f"Response: {method} {path} - {status_code} ({latency_ms:.2f}ms)",
            extra={"extra_data": log_data},
        )

    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        latency_ms: float,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Log a prediction result.

        Args:
            prediction: Predicted class
            confidence: Confidence score
            latency_ms: Inference latency
            request_id: Unique request identifier
        """
        log_data = {
            "event": "prediction_made",
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "inference_time_ms": round(latency_ms, 2),
            "request_id": request_id,
        }

        self.logger.info(
            f"Prediction: {prediction} (confidence: {confidence:.4f})",
            extra={"extra_data": log_data},
        )


# Global request logger instance
request_logger = RequestLogger()
