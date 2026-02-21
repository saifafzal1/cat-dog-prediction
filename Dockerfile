# Dockerfile for Cats vs Dogs Classification Inference Service
# This Dockerfile creates a containerized REST API for model inference

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set the model path environment variable
ENV MODEL_PATH=/app/models/best_model.pt
# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Install system dependencies
# These are required for Pillow and other image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Install heavy ML libraries first for better caching
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check to verify container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Command to run the application
# Using uvicorn with production settings
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
