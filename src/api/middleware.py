"""
FastAPI middleware for request logging and metrics collection.

This module provides middleware components that automatically
track requests, responses, and collect performance metrics.
"""

import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

from src.monitoring.metrics import record_request
from src.monitoring.logging_config import request_logger

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all HTTP requests and responses.

    Automatically logs:
    - Incoming requests with method, path, and client info
    - Outgoing responses with status code and latency
    - Excludes sensitive data from logs
    """

    # Paths to exclude from detailed logging
    EXCLUDED_PATHS = {"/health", "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain

        Returns:
            The response from the handler
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Get client IP (anonymized in logger)
        client_ip = request.client.host if request.client else None

        # Skip detailed logging for excluded paths
        should_log = request.url.path not in self.EXCLUDED_PATHS

        # Log incoming request
        if should_log:
            request_logger.log_request(
                method=request.method,
                path=request.url.path,
                client_ip=client_ip,
                request_id=request_id,
            )

        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            success = response.status_code < 400
        except Exception as e:
            # Log exception
            logger.error(f"Request failed with exception: {e}", exc_info=True)
            raise

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Record metrics
        record_request(endpoint=request.url.path, latency_ms=latency_ms, success=success)

        # Log response
        if should_log:
            request_logger.log_response(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request_id=request_id,
            )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting detailed metrics.

    Collects metrics on request rates, latencies, and error rates
    for monitoring and alerting purposes.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and collect metrics.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain

        Returns:
            The response from the handler
        """
        start_time = time.time()

        response = await call_next(request)

        latency_ms = (time.time() - start_time) * 1000

        # Record metrics (already done in logging middleware, but can add more here)
        # This is a placeholder for additional metrics collection

        return response
