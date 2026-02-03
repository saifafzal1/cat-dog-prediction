# Execution Guide

This guide provides step-by-step instructions to run the complete MLOps pipeline for Cats vs Dogs Classification.

## Prerequisites

- Python 3.9+ installed
- Docker installed and running
- Git installed
- Kaggle account (for dataset download)

## Step 1: Initial Setup

### 1.1 Clone/Navigate to Project

```bash
cd "/Users/nadiaashfaq/Documents/MLOPS/MLOPS Assignment 2"
```

### 1.2 Create and Activate Virtual Environment

```bash
# Create virtual environment (if not exists)
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.4 Initialize Git and DVC

```bash
# Initialize Git (if not already done)
git init
git branch -m main

# Initialize DVC
dvc init
```

## Step 2: Configure Kaggle API (Dataset Download)

### 2.1 Get Kaggle API Credentials

1. Go to https://www.kaggle.com
2. Login or create an account
3. Go to Account Settings (click profile icon -> Settings)
4. Scroll to "API" section
5. Click "Create New Token"
6. Download `kaggle.json`

### 2.2 Setup Kaggle Credentials

```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Move kaggle.json to the directory
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## Step 3: Run the ML Pipeline (M1)

### 3.1 Download Dataset

```bash
python src/data/download.py
```

### 3.2 Preprocess Data

```bash
python src/data/preprocess.py
```

### 3.3 Train Model with MLflow Tracking

```bash
python src/train.py
```

### 3.4 View Experiments in MLflow

```bash
# Start MLflow UI (in a new terminal)
mlflow ui --port 5001
```

Open http://localhost:5001 in your browser to view experiments.

### Alternative: Run Entire Pipeline with DVC

```bash
dvc repro
```

## Step 4: Run the API Service (M2)

### 4.1 Start the API Server

```bash
# Development mode with hot reload
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or using Makefile
make api-dev
```

### 4.2 Test API Endpoints

Open a new terminal and run:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
# Open in browser: http://localhost:8000/docs

# Test prediction (replace with actual image path)
curl -X POST -F "file=@/path/to/cat_or_dog_image.jpg" http://localhost:8000/predict
```

## Step 5: Build and Run Docker Container (M2)

### 5.1 Build Docker Image

```bash
docker build -t cats-dogs-classifier:latest .

# Or using Makefile
make docker-build
```

### 5.2 Run Container

```bash
docker run -d \
  --name cats-dogs-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  cats-dogs-classifier:latest

# Or using Makefile
make docker-run
```

### 5.3 Verify Container

```bash
# Check container is running
docker ps

# Test health endpoint
curl http://localhost:8000/health

# View logs
docker logs cats-dogs-api
```

### 5.4 Stop Container

```bash
docker stop cats-dogs-api
docker rm cats-dogs-api

# Or using Makefile
make docker-stop
```

## Step 6: Run Tests (M3)

### 6.1 Run All Unit Tests

```bash
pytest tests/ -v

# Or using Makefile
make test
```

### 6.2 Run Tests with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html

# Or using Makefile
make test-cov
```

### 6.3 Run Linting

```bash
# Check code formatting
black --check src/ tests/

# Check imports
isort --check-only src/ tests/

# Run flake8
flake8 src/ tests/

# Or all at once using Makefile
make lint
```

## Step 7: Deploy with Docker Compose (M4)

### 7.1 Start Services

```bash
docker-compose up -d

# Or using deployment script
./scripts/deploy.sh compose-up
```

### 7.2 Run Smoke Tests

```bash
./scripts/smoke_test.sh http://localhost:8000

# Or using Makefile
make smoke-test
```

### 7.3 Stop Services

```bash
docker-compose down

# Or using deployment script
./scripts/deploy.sh compose-down
```

## Step 8: Deploy to Kubernetes (M4 - Optional)

### 8.1 Start Local Kubernetes (using minikube)

```bash
# Install minikube if not installed
brew install minikube  # On macOS

# Start minikube
minikube start

# Enable ingress
minikube addons enable ingress
```

### 8.2 Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -k deploy/kubernetes/

# Or using deployment script
./scripts/deploy.sh k8s-deploy
```

### 8.3 Check Deployment Status

```bash
kubectl get all -n cats-dogs-classifier

# Or using deployment script
./scripts/deploy.sh k8s-status
```

### 8.4 Access the Service

```bash
# Get service URL
minikube service cats-dogs-api-nodeport -n cats-dogs-classifier --url
```

### 8.5 Delete Deployment

```bash
kubectl delete -k deploy/kubernetes/

# Or using deployment script
./scripts/deploy.sh k8s-delete
```

## Step 9: Monitor the Service (M5)

### 9.1 View Metrics

```bash
curl http://localhost:8000/metrics | python -m json.tool
```

### 9.2 View Model Performance

```bash
curl http://localhost:8000/performance | python -m json.tool
```

### 9.3 View Recent Predictions

```bash
curl "http://localhost:8000/predictions/recent?n=10" | python -m json.tool
```

### 9.4 Simulate Predictions for Testing

```bash
# Simulate 20 predictions
python scripts/simulate_predictions.py --url http://localhost:8000 -n 20

# Simulate with labels for accuracy tracking
python scripts/simulate_predictions.py --url http://localhost:8000 -n 20 --add-labels

# View metrics only
python scripts/simulate_predictions.py --url http://localhost:8000 --metrics-only
```

### 9.5 Add True Label to Prediction

```bash
# Add label to most recent prediction
curl -X POST "http://localhost:8000/predictions/-1/label?true_label=cat"
```

## Step 10: CI/CD Pipeline (M3 & M4)

The CI/CD pipeline runs automatically on GitHub when you push code.

### 10.1 Push to GitHub

```bash
# Add remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/mlops-assignment-2.git

# Add all files
git add .

# Commit
git commit -m "Complete MLOps pipeline implementation"

# Push
git push -u origin main
```

### 10.2 Configure GitHub Secrets

Go to your GitHub repository -> Settings -> Secrets and variables -> Actions

Add the following secrets:
- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token

### 10.3 View CI/CD Pipeline

Go to your GitHub repository -> Actions tab to view pipeline runs.

## Quick Command Reference

```bash
# Setup
make install              # Install dependencies
source venv/bin/activate  # Activate environment

# ML Pipeline
python src/data/download.py      # Download dataset
python src/data/preprocess.py    # Preprocess data
python src/train.py              # Train model
dvc repro                        # Run full DVC pipeline
mlflow ui --port 5000            # View experiments

# API
make api-dev              # Start API (dev mode)
make docker-build         # Build Docker image
make docker-run           # Run container
make docker-test          # Test container

# Testing
make test                 # Run all tests
make lint                 # Run linting
make test-cov             # Test with coverage

# Deployment
make deploy-compose       # Docker Compose deployment
make deploy-k8s           # Kubernetes deployment
make smoke-test           # Run smoke tests

# Monitoring
curl http://localhost:8000/metrics
curl http://localhost:8000/performance
```

## Troubleshooting

### Issue: Model not found

Ensure you have trained the model first:
```bash
python src/train.py
```

### Issue: Kaggle download fails

Verify kaggle.json is properly configured:
```bash
cat ~/.kaggle/kaggle.json
```

### Issue: Docker build fails

Ensure Docker daemon is running:
```bash
docker info
```

### Issue: Port already in use

Stop existing services:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

### Issue: Tests fail

Ensure numpy version is compatible:
```bash
pip install "numpy<2"
```
