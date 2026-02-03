#!/bin/bash
# Script to build and run the Docker container for the inference service
# Usage: ./scripts/docker_build_run.sh [build|run|stop|logs|test]

set -e

# Configuration
IMAGE_NAME="cats-dogs-classifier"
IMAGE_TAG="latest"
CONTAINER_NAME="cats-dogs-api"
PORT=8000

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build the Docker image
build() {
    print_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    print_info "Build completed successfully"
}

# Run the Docker container
run() {
    # Check if container is already running
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_warn "Container ${CONTAINER_NAME} is already running"
        print_info "Use './scripts/docker_build_run.sh stop' to stop it first"
        return 1
    fi

    # Remove existing stopped container if exists
    if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        print_info "Removing existing stopped container"
        docker rm ${CONTAINER_NAME}
    fi

    print_info "Starting container: ${CONTAINER_NAME}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8000 \
        -v "$(pwd)/models:/app/models:ro" \
        ${IMAGE_NAME}:${IMAGE_TAG}

    print_info "Container started successfully"
    print_info "API available at: http://localhost:${PORT}"
    print_info "Health check: http://localhost:${PORT}/health"
    print_info "API docs: http://localhost:${PORT}/docs"
}

# Stop the Docker container
stop() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        print_info "Stopping container: ${CONTAINER_NAME}"
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        print_info "Container stopped and removed"
    else
        print_warn "Container ${CONTAINER_NAME} is not running"
    fi
}

# View container logs
logs() {
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "Container ${CONTAINER_NAME} is not running"
        return 1
    fi
}

# Test the API endpoints
test_api() {
    print_info "Testing API endpoints..."

    # Wait for container to be ready
    print_info "Waiting for service to be ready..."
    sleep 3

    # Test health endpoint
    print_info "Testing /health endpoint..."
    curl -s http://localhost:${PORT}/health | python3 -m json.tool

    echo ""

    # Test root endpoint
    print_info "Testing / endpoint..."
    curl -s http://localhost:${PORT}/ | python3 -m json.tool

    echo ""

    # Test model info endpoint
    print_info "Testing /model/info endpoint..."
    curl -s http://localhost:${PORT}/model/info | python3 -m json.tool

    echo ""
    print_info "API tests completed"
}

# Test prediction with a sample image
test_predict() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 test_predict <image_path>"
        return 1
    fi

    if [ ! -f "$1" ]; then
        print_error "Image file not found: $1"
        return 1
    fi

    print_info "Testing /predict endpoint with image: $1"
    curl -X POST \
        -F "file=@$1" \
        http://localhost:${PORT}/predict | python3 -m json.tool
}

# Show help
help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build         Build the Docker image"
    echo "  run           Run the Docker container"
    echo "  stop          Stop and remove the container"
    echo "  logs          View container logs"
    echo "  test          Test API endpoints"
    echo "  test_predict  Test prediction with an image"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 test_predict path/to/image.jpg"
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    run)
        run
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    test)
        test_api
        ;;
    test_predict)
        test_predict "$2"
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        help
        exit 1
        ;;
esac
