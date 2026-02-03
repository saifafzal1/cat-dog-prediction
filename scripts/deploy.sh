#!/bin/bash
# Deployment helper script for Cats vs Dogs Classification API
# Supports Docker Compose and Kubernetes deployments
#
# Usage: ./scripts/deploy.sh [command] [options]
# Commands:
#   compose-up      Start services with Docker Compose
#   compose-down    Stop Docker Compose services
#   k8s-deploy      Deploy to Kubernetes
#   k8s-delete      Delete Kubernetes deployment
#   smoke-test      Run smoke tests

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_IMAGE="cats-dogs-classifier"
DOCKER_TAG="${IMAGE_TAG:-latest}"
K8S_NAMESPACE="cats-dogs-classifier"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Docker Compose deployment
compose_up() {
    print_info "Starting services with Docker Compose..."

    cd "$PROJECT_ROOT"

    # Build image if it does not exist
    if ! docker image inspect "${DOCKER_IMAGE}:${DOCKER_TAG}" > /dev/null 2>&1; then
        print_info "Building Docker image..."
        docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" .
    fi

    # Start services
    docker-compose up -d

    print_info "Waiting for services to be ready..."
    sleep 5

    # Run smoke tests
    if ./scripts/smoke_test.sh http://localhost:8000; then
        print_success "Deployment successful!"
        print_info "API available at: http://localhost:8000"
        print_info "Health check: http://localhost:8000/health"
        print_info "API docs: http://localhost:8000/docs"
    else
        print_error "Smoke tests failed!"
        print_warn "Check logs with: docker-compose logs"
        exit 1
    fi
}

compose_down() {
    print_info "Stopping Docker Compose services..."

    cd "$PROJECT_ROOT"
    docker-compose down

    print_success "Services stopped"
}

compose_logs() {
    cd "$PROJECT_ROOT"
    docker-compose logs -f
}

# Kubernetes deployment
k8s_deploy() {
    print_info "Deploying to Kubernetes..."

    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info > /dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi

    cd "$PROJECT_ROOT"

    # Update image tag in kustomization
    if [ -n "$IMAGE_TAG" ]; then
        print_info "Using image tag: $IMAGE_TAG"
        sed -i.bak "s/newTag: .*/newTag: $IMAGE_TAG/" deploy/kubernetes/kustomization.yaml
    fi

    # Apply Kubernetes manifests
    print_info "Applying Kubernetes manifests..."
    kubectl apply -k deploy/kubernetes/

    # Wait for deployment
    print_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/cats-dogs-api -n $K8S_NAMESPACE --timeout=300s

    # Get service URL
    NODE_PORT=$(kubectl get svc cats-dogs-api-nodeport -n $K8S_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    print_success "Deployment successful!"
    print_info "Service available at NodePort: $NODE_PORT"

    # Run smoke tests
    print_info "Running smoke tests..."
    ./scripts/smoke_test.sh "http://localhost:$NODE_PORT" || true
}

k8s_delete() {
    print_info "Deleting Kubernetes deployment..."

    cd "$PROJECT_ROOT"
    kubectl delete -k deploy/kubernetes/ --ignore-not-found

    print_success "Kubernetes deployment deleted"
}

k8s_status() {
    print_info "Kubernetes deployment status:"
    echo ""

    kubectl get all -n $K8S_NAMESPACE 2>/dev/null || print_warn "Namespace not found"
}

# Run smoke tests
run_smoke_test() {
    local url="${1:-http://localhost:8000}"
    print_info "Running smoke tests against $url..."

    "$SCRIPT_DIR/smoke_test.sh" "$url"
}

# Show help
show_help() {
    echo "Deployment Helper Script for Cats vs Dogs Classification API"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  compose-up      Start services with Docker Compose"
    echo "  compose-down    Stop Docker Compose services"
    echo "  compose-logs    View Docker Compose logs"
    echo "  k8s-deploy      Deploy to Kubernetes cluster"
    echo "  k8s-delete      Delete Kubernetes deployment"
    echo "  k8s-status      Show Kubernetes deployment status"
    echo "  smoke-test      Run smoke tests (default: http://localhost:8000)"
    echo "  help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  IMAGE_TAG       Docker image tag to deploy (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 compose-up"
    echo "  $0 k8s-deploy"
    echo "  IMAGE_TAG=v1.0.0 $0 k8s-deploy"
    echo "  $0 smoke-test http://localhost:8000"
}

# Main script logic
case "${1:-help}" in
    compose-up)
        compose_up
        ;;
    compose-down)
        compose_down
        ;;
    compose-logs)
        compose_logs
        ;;
    k8s-deploy)
        k8s_deploy
        ;;
    k8s-delete)
        k8s_delete
        ;;
    k8s-status)
        k8s_status
        ;;
    smoke-test)
        run_smoke_test "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
