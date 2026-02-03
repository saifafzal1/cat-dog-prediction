# Makefile for Cats vs Dogs Classification MLOps Project
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test lint format clean docker-build docker-run docker-stop train api deploy-compose deploy-compose-down deploy-k8s deploy-k8s-delete smoke-test

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install production dependencies"
	@echo "    make install-dev    Install development dependencies"
	@echo ""
	@echo "  Development:"
	@echo "    make lint           Run linting checks"
	@echo "    make format         Format code with Black and isort"
	@echo "    make test           Run unit tests"
	@echo "    make test-cov       Run tests with coverage report"
	@echo ""
	@echo "  ML Pipeline:"
	@echo "    make train          Run model training"
	@echo "    make dvc-repro      Reproduce DVC pipeline"
	@echo ""
	@echo "  API:"
	@echo "    make api            Start the FastAPI server"
	@echo "    make api-dev        Start API in development mode"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build   Build Docker image"
	@echo "    make docker-run     Run Docker container"
	@echo "    make docker-stop    Stop Docker container"
	@echo "    make docker-test    Test Docker container"
	@echo ""
	@echo "  Deployment:"
	@echo "    make deploy-compose      Deploy with Docker Compose"
	@echo "    make deploy-compose-down Stop Docker Compose deployment"
	@echo "    make deploy-k8s          Deploy to Kubernetes"
	@echo "    make deploy-k8s-delete   Delete Kubernetes deployment"
	@echo "    make smoke-test          Run smoke tests"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean          Remove build artifacts"
	@echo "    make clean-all      Remove all generated files"

# Configuration
PYTHON := python
PIP := pip
DOCKER_IMAGE := cats-dogs-classifier
DOCKER_TAG := latest
CONTAINER_NAME := cats-dogs-api
PORT := 8000

# Setup commands
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install black isort flake8 pytest-cov

# Development commands
lint:
	@echo "Running Flake8..."
	flake8 src/ tests/
	@echo "Checking Black formatting..."
	black --check src/ tests/
	@echo "Checking isort..."
	isort --check-only src/ tests/
	@echo "All lint checks passed!"

format:
	@echo "Formatting with Black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort src/ tests/
	@echo "Formatting complete!"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/"

# ML Pipeline commands
train:
	$(PYTHON) src/train.py

dvc-repro:
	dvc repro

# API commands
api:
	uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT)

api-dev:
	uvicorn src.api.app:app --host 0.0.0.0 --port $(PORT) --reload

# Docker commands
docker-build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v $(PWD)/models:/app/models:ro \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "Container started. API available at http://localhost:$(PORT)"

docker-stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	@echo "Container stopped and removed"

docker-test:
	@echo "Testing health endpoint..."
	curl -s http://localhost:$(PORT)/health | python -m json.tool
	@echo ""
	@echo "Testing root endpoint..."
	curl -s http://localhost:$(PORT)/ | python -m json.tool

docker-logs:
	docker logs -f $(CONTAINER_NAME)

# Deployment commands
deploy-compose:
	./scripts/deploy.sh compose-up

deploy-compose-down:
	./scripts/deploy.sh compose-down

deploy-k8s:
	./scripts/deploy.sh k8s-deploy

deploy-k8s-delete:
	./scripts/deploy.sh k8s-delete

smoke-test:
	./scripts/smoke_test.sh http://localhost:$(PORT)

# Cleanup commands
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml
	@echo "Cleaned build artifacts"

clean-all: clean
	rm -rf mlruns/ mlartifacts/
	rm -rf data/raw/* data/processed/*
	rm -rf models/*.pt models/*.pkl
	@echo "Cleaned all generated files"
