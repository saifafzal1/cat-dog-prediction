# Screen Recording Guide

This guide provides a script for creating the required 5-minute screen recording demonstrating the complete MLOps workflow.

## Recording Setup

### Recommended Tools

- **macOS**: QuickTime Player (built-in) or OBS Studio
- **Windows**: OBS Studio or Xbox Game Bar
- **Linux**: OBS Studio or SimpleScreenRecorder

### Recording Settings

- Resolution: 1920x1080 (or your screen resolution)
- Frame Rate: 30 fps
- Format: MP4
- Duration: Less than 5 minutes

## Recording Script (Follow This Order)

### Section 1: Project Overview (30 seconds)

```
1. Open terminal in project directory
2. Show project structure:
   ls -la
   find . -type d -not -path "./venv/*" -not -path "./.git/*" | head -20

3. Briefly explain: "This is the MLOps pipeline for Cats vs Dogs classification"
```

**Say**: "This is the complete MLOps pipeline with model development, containerization, CI/CD, and monitoring."

---

### Section 2: Model Training with MLflow (1 minute)

```bash
# Activate environment
source venv/bin/activate

# Show training script briefly
head -50 src/train.py

# Start MLflow UI in background
mlflow ui --port 5000 &

# If model already trained, show the MLflow UI
# Open browser: http://localhost:5000
```

**Say**: "The training script uses MLflow for experiment tracking. Here you can see logged parameters, metrics, and artifacts including confusion matrix and loss curves."

**Show in MLflow UI**:
- Experiment list
- Run parameters
- Metrics (accuracy, loss)
- Artifacts (confusion matrix image)

---

### Section 3: API Service Demo (1 minute)

```bash
# Start the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

# Wait for startup
sleep 5

# Test health endpoint
curl http://localhost:8000/health | python -m json.tool

# Show API documentation
# Open browser: http://localhost:8000/docs
```

**Say**: "The FastAPI service provides endpoints for health checks, predictions, and monitoring."

**Show in browser**:
- Swagger UI at /docs
- Available endpoints
- Try the /predict endpoint with a test image

---

### Section 4: Docker Containerization (45 seconds)

```bash
# Build Docker image
docker build -t cats-dogs-classifier:latest .

# Show image
docker images | grep cats-dogs

# Run container
docker run -d --name demo-api -p 8001:8000 -v $(pwd)/models:/app/models:ro cats-dogs-classifier:latest

# Test containerized API
curl http://localhost:8001/health | python -m json.tool

# Stop container
docker stop demo-api && docker rm demo-api
```

**Say**: "The application is containerized with Docker for consistent deployment across environments."

---

### Section 5: CI/CD Pipeline (45 seconds)

```bash
# Show CI workflow
cat .github/workflows/ci.yaml | head -60

# Show CD workflow
cat .github/workflows/cd.yaml | head -40
```

**If you have GitHub access, show**:
- GitHub Actions tab
- Recent workflow runs
- Pipeline stages (lint, test, build)

**Say**: "GitHub Actions automates testing, building, and deployment. On every push, the pipeline runs linting, unit tests, builds the Docker image, and pushes to the container registry."

---

### Section 6: Deployment Demo (45 seconds)

```bash
# Deploy with Docker Compose
docker-compose up -d

# Show running containers
docker-compose ps

# Run smoke tests
./scripts/smoke_test.sh http://localhost:8000
```

**Say**: "Deployment can be done via Docker Compose for local/staging or Kubernetes for production. Smoke tests automatically verify the deployment."

---

### Section 7: Monitoring Demo (45 seconds)

```bash
# Show metrics
curl http://localhost:8000/metrics | python -m json.tool

# Simulate some predictions
python scripts/simulate_predictions.py -n 5 --add-labels

# Show performance metrics
curl http://localhost:8000/performance | python -m json.tool

# Show recent predictions
curl "http://localhost:8000/predictions/recent?n=5" | python -m json.tool
```

**Say**: "The monitoring system tracks request metrics, latencies, prediction distribution, and model accuracy when true labels are provided."

---

### Section 8: Cleanup and Summary (15 seconds)

```bash
# Stop all services
docker-compose down
pkill -f "uvicorn"
pkill -f "mlflow"

# Show test results
pytest tests/ -v --tb=no | tail -10
```

**Say**: "All 77 unit tests pass. This completes the demonstration of the end-to-end MLOps pipeline."

---

## Quick Demo Commands (Copy-Paste Ready)

```bash
# === PREPARATION (run before recording) ===
cd "/Users/nadiaashfaq/Documents/MLOPS/MLOPS Assignment 2"
source venv/bin/activate
docker-compose down 2>/dev/null
pkill -f uvicorn 2>/dev/null
pkill -f mlflow 2>/dev/null

# === DURING RECORDING ===

# 1. Project Overview
clear
echo "=== MLOps Cats vs Dogs Classification ==="
ls -la
echo ""
echo "=== Project Structure ==="
find . -type d -not -path "./venv/*" -not -path "./.git/*" -not -path "./.dvc/*" | head -15

# 2. MLflow (if model trained)
mlflow ui --port 5000 &
sleep 3
echo "MLflow UI: http://localhost:5000"

# 3. Start API
uvicorn src.api.app:app --port 8000 &
sleep 5
echo ""
echo "=== Health Check ==="
curl -s http://localhost:8000/health | python -m json.tool
echo ""
echo "API Docs: http://localhost:8000/docs"

# 4. Docker Build & Run
echo ""
echo "=== Docker Build ==="
docker build -t cats-dogs-classifier:demo . 2>&1 | tail -5
echo ""
docker images | grep cats-dogs

# 5. Show CI/CD
echo ""
echo "=== CI Pipeline (GitHub Actions) ==="
head -30 .github/workflows/ci.yaml

# 6. Docker Compose Deployment
echo ""
echo "=== Deployment ==="
docker-compose up -d
sleep 5
docker-compose ps

# 7. Smoke Tests
echo ""
echo "=== Smoke Tests ==="
./scripts/smoke_test.sh http://localhost:8000

# 8. Monitoring
echo ""
echo "=== Metrics ==="
curl -s http://localhost:8000/metrics | python -m json.tool

echo ""
echo "=== Simulating Predictions ==="
python scripts/simulate_predictions.py -n 3 --add-labels 2>/dev/null

echo ""
echo "=== Performance Metrics ==="
curl -s http://localhost:8000/performance | python -m json.tool

# 9. Tests
echo ""
echo "=== Unit Tests ==="
pytest tests/ --tb=no -q

# 10. Cleanup
echo ""
echo "=== Cleanup ==="
docker-compose down
pkill -f uvicorn
pkill -f mlflow
echo "Demo complete!"
```

## Recording Tips

1. **Before Recording**:
   - Close unnecessary applications
   - Increase terminal font size for readability
   - Have all commands ready in a text file
   - Do a practice run

2. **During Recording**:
   - Speak clearly and at a moderate pace
   - Pause briefly after each command to show output
   - Highlight important outputs verbally
   - Keep within 5 minutes

3. **After Recording**:
   - Trim any dead time
   - Add captions if needed
   - Export in MP4 format
   - Compress if file is too large

## Checklist for Recording

- [ ] Show project structure
- [ ] Demonstrate MLflow experiment tracking
- [ ] Show API endpoints (health, predict, docs)
- [ ] Build and run Docker container
- [ ] Show CI/CD workflow configuration
- [ ] Deploy with Docker Compose
- [ ] Run smoke tests
- [ ] Show monitoring metrics
- [ ] Run unit tests
- [ ] Total duration < 5 minutes

## File Submission

After recording:
1. Save video as `mlops_demo.mp4`
2. If file > 100MB, upload to Google Drive/YouTube and share link
3. Include in submission zip or provide link in README
