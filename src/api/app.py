"""
FastAPI inference service for Cats vs Dogs classification.

This module provides a REST API for serving predictions from the
trained model. It includes health check, prediction, and monitoring endpoints.
"""

import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from PIL import Image
from pydantic import BaseModel, Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import get_eval_transforms  # noqa: E402
from src.models.cnn import SimpleCNN  # noqa: E402
from src.monitoring.logging_config import request_logger, setup_logging  # noqa: E402
from src.monitoring.metrics import get_metrics, record_prediction  # noqa: E402
from src.monitoring.performance_tracker import performance_tracker  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Set up logging
setup_logging(
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    log_format="standard",
    log_file=os.environ.get("LOG_FILE", "logs/api.log"),
)
logger = logging.getLogger(__name__)


# Global variables for model and transforms
model: Optional[SimpleCNN] = None
transforms = None
device: Optional[torch.device] = None
config: Optional[Dict[str, Any]] = None

# Class labels mapping
CLASS_NAMES = {0: "cat", 1: "dog"}


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Compute device being used")
    version: str = Field(..., description="API version")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence score for prediction")
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class ErrorResponse(BaseModel):
    """Response model for error cases."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


def load_model_weights(model_path: str) -> SimpleCNN:
    """
    Load the trained model weights from a checkpoint file.

    Args:
        model_path: Path to the model checkpoint file (.pt)

    Returns:
        Loaded SimpleCNN model

    Raises:
        FileNotFoundError: If model file does not exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Get model configuration from checkpoint or use defaults
    model_config = checkpoint.get("config", {}).get("model", {})
    num_classes = model_config.get("num_classes", 2)
    dropout_rate = model_config.get("dropout_rate", 0.5)

    # Create model instance
    model = SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("Model loaded successfully")
    return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocess an image for model inference.

    The image is opened, converted to RGB, resized to 224x224,
    and normalized using ImageNet statistics.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed image tensor of shape (1, 3, 224, 224)

    Raises:
        ValueError: If image cannot be processed
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (handles grayscale, RGBA, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to expected input size
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        # Apply transforms (ToTensor + Normalize)
        if transforms is not None:
            tensor = transforms(image)
        else:
            # Manual conversion without torchvision transforms
            # This avoids potential numpy compatibility issues
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Transpose from HWC to CHW format
            img_array = img_array.transpose(2, 0, 1)

            # Convert to tensor
            tensor = torch.from_numpy(img_array)

            # Normalize with ImageNet statistics
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events, including model loading.
    """
    global model, transforms, device, config

    logger.info("Starting up inference service...")

    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults")
        config = {}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load transforms
    transforms = get_eval_transforms(config)

    # Load model
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pt")

    try:
        model = load_model_weights(model_path)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded and ready for inference")
    except FileNotFoundError:
        logger.warning(
            f"Model not found at {model_path}. "
            "Service will start but predictions will fail until model is available."
        )
        model = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

    yield

    # Cleanup on shutdown
    logger.info("Shutting down inference service...")
    model = None


# Create FastAPI application
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="REST API for binary image classification (Cats vs Dogs)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the inference service",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the service including whether
    the model is loaded and ready for predictions.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        version="1.0.0",
    )


@app.get("/", summary="Root Endpoint", description="Welcome message and API information")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.
    """
    return {
        "message": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict"},
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Predict Image Class",
    description="Upload an image to get cat/dog classification prediction",
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)")
) -> PredictionResponse:
    """
    Prediction endpoint.

    Accepts an image file and returns the predicted class (cat or dog)
    along with confidence scores and probabilities.

    Args:
        file: Uploaded image file

    Returns:
        PredictionResponse with classification results

    Raises:
        HTTPException: If model not loaded, invalid image, or inference fails
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure model file exists.",
        )

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. " "Supported types: JPEG, PNG",
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded"
            )

        # Preprocess image
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(device)

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get prediction
        probs = probabilities[0].cpu().numpy()
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # Build probability dictionary
        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        # Log prediction
        request_logger.log_prediction(
            prediction=CLASS_NAMES[predicted_class],
            confidence=confidence,
            latency_ms=inference_time,
        )

        # Record metrics
        record_prediction(CLASS_NAMES[predicted_class], confidence)

        # Track for performance monitoring
        performance_tracker.record_prediction(
            prediction=CLASS_NAMES[predicted_class], confidence=confidence, probabilities=prob_dict
        )

        logger.info(
            f"Prediction: {CLASS_NAMES[predicted_class]} "
            f"(confidence: {confidence:.4f}, time: {inference_time:.2f}ms)"
        )

        return PredictionResponse(
            prediction=CLASS_NAMES[predicted_class],
            confidence=confidence,
            probabilities=prob_dict,
            inference_time_ms=round(inference_time, 2),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model/info", summary="Model Information", description="Get information about the loaded model"
)
async def model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.
    """
    if model is None:
        return {"loaded": False, "message": "No model loaded"}

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "loaded": True,
        "model_name": "SimpleCNN",
        "num_classes": model.num_classes,
        "dropout_rate": model.dropout_rate,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "input_shape": [1, 3, 224, 224],
        "class_labels": CLASS_NAMES,
    }


@app.get(
    "/metrics",
    summary="Service Metrics",
    description="Get service performance metrics including request counts and latencies",
)
async def metrics() -> Dict[str, Any]:
    """
    Metrics endpoint for monitoring.

    Returns request counts, latencies, and prediction statistics.
    """
    return get_metrics()


@app.get(
    "/performance",
    summary="Model Performance",
    description="Get model performance metrics including accuracy and prediction distribution",
)
async def performance() -> Dict[str, Any]:
    """
    Model performance endpoint.

    Returns prediction distribution, confidence statistics,
    and accuracy metrics (when labels are available).
    """
    return performance_tracker.get_performance_metrics()


@app.get(
    "/predictions/recent",
    summary="Recent Predictions",
    description="Get the most recent prediction records",
)
async def recent_predictions(
    n: int = Query(default=10, ge=1, le=100, description="Number of predictions to return")
) -> List[Dict[str, Any]]:
    """
    Get recent predictions.

    Args:
        n: Number of recent predictions to return (1-100)

    Returns:
        List of recent prediction records
    """
    return performance_tracker.get_recent_predictions(n)


@app.post(
    "/predictions/{index}/label",
    summary="Add True Label",
    description="Add a true label to a prediction for accuracy tracking",
)
async def add_label(
    index: int, true_label: str = Query(..., description="True label (cat or dog)")
) -> Dict[str, str]:
    """
    Add a true label to a prediction record.

    This allows tracking model accuracy when ground truth becomes available.

    Args:
        index: Index of the prediction (use negative for recent, e.g., -1 for last)
        true_label: The correct label

    Returns:
        Confirmation message
    """
    if true_label not in ["cat", "dog"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Label must be 'cat' or 'dog'"
        )

    success = performance_tracker.add_true_label(index, true_label)

    if success:
        return {"message": f"Label '{true_label}' added to prediction at index {index}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Prediction at index {index} not found"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
