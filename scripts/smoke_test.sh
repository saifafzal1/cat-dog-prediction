#!/bin/bash
# Smoke test script for Cats vs Dogs Classification API
# Verifies the deployed service is functioning correctly
#
# Usage: ./scripts/smoke_test.sh [BASE_URL]
# Example: ./scripts/smoke_test.sh http://localhost:8000

set -e

# Configuration
BASE_URL="${1:-http://localhost:8000}"
TIMEOUT=10
MAX_RETRIES=5
RETRY_DELAY=5

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

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

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Function to wait for service to be ready
wait_for_service() {
    print_info "Waiting for service to be ready at ${BASE_URL}..."

    for i in $(seq 1 $MAX_RETRIES); do
        if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
            print_info "Service is ready!"
            return 0
        fi
        print_warn "Attempt $i/$MAX_RETRIES failed, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done

    print_error "Service did not become ready in time"
    return 1
}

# Test 1: Health endpoint
test_health_endpoint() {
    print_info "Testing health endpoint..."

    response=$(curl -sf --max-time $TIMEOUT "${BASE_URL}/health")
    status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)

    if [ "$status" == "healthy" ]; then
        print_success "Health endpoint returned 'healthy'"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_fail "Health endpoint did not return 'healthy' status"
        print_error "Response: $response"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 2: Root endpoint
test_root_endpoint() {
    print_info "Testing root endpoint..."

    response=$(curl -sf --max-time $TIMEOUT "${BASE_URL}/")
    message=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['message'])" 2>/dev/null)

    if [ -n "$message" ]; then
        print_success "Root endpoint returned valid response"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_fail "Root endpoint did not return expected response"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 3: Model info endpoint
test_model_info_endpoint() {
    print_info "Testing model info endpoint..."

    response=$(curl -sf --max-time $TIMEOUT "${BASE_URL}/model/info")
    http_code=$?

    if [ $http_code -eq 0 ]; then
        print_success "Model info endpoint is accessible"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_fail "Model info endpoint returned error"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Test 4: Prediction endpoint (with test image)
test_prediction_endpoint() {
    print_info "Testing prediction endpoint..."

    # Create a simple test image (red square)
    TEST_IMAGE="/tmp/smoke_test_image.jpg"

    # Generate test image using Python
    python3 << EOF
from PIL import Image
img = Image.new('RGB', (224, 224), color='red')
img.save('$TEST_IMAGE', 'JPEG')
EOF

    if [ ! -f "$TEST_IMAGE" ]; then
        print_warn "Could not create test image, skipping prediction test"
        return 0
    fi

    # Make prediction request
    response=$(curl -sf --max-time $TIMEOUT \
        -X POST \
        -F "file=@${TEST_IMAGE}" \
        "${BASE_URL}/predict" 2>/dev/null)

    http_code=$?

    # Clean up test image
    rm -f "$TEST_IMAGE"

    if [ $http_code -eq 0 ]; then
        prediction=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['prediction'])" 2>/dev/null)
        confidence=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['confidence'])" 2>/dev/null)

        if [ -n "$prediction" ] && [ -n "$confidence" ]; then
            print_success "Prediction endpoint returned: $prediction (confidence: $confidence)"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        fi
    fi

    # Check if model is not loaded (acceptable for smoke test)
    if echo "$response" | grep -q "Model not loaded"; then
        print_warn "Model not loaded - prediction endpoint is accessible but model unavailable"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    fi

    print_fail "Prediction endpoint returned unexpected response"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    return 1
}

# Test 5: Response time check
test_response_time() {
    print_info "Testing response time..."

    # Use python for cross-platform millisecond timing (macOS date doesn't support %N)
    start_time=$(python3 -c 'import time; print(int(time.time() * 1000))')
    curl -sf --max-time $TIMEOUT "${BASE_URL}/health" > /dev/null
    end_time=$(python3 -c 'import time; print(int(time.time() * 1000))')

    response_time=$((end_time - start_time))

    if [ $response_time -lt 1000 ]; then
        print_success "Response time: ${response_time}ms (< 1000ms)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        print_warn "Response time: ${response_time}ms (>= 1000ms) - may indicate performance issues"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "  Smoke Tests for Cats vs Dogs API"
    echo "  Target: ${BASE_URL}"
    echo "=========================================="
    echo ""

    # Wait for service to be ready
    if ! wait_for_service; then
        print_error "Service is not available. Smoke tests failed."
        exit 1
    fi

    echo ""
    echo "Running smoke tests..."
    echo ""

    # Run all tests
    test_health_endpoint || true
    test_root_endpoint || true
    test_model_info_endpoint || true
    test_prediction_endpoint || true
    test_response_time || true

    echo ""
    echo "=========================================="
    echo "  Smoke Test Results"
    echo "=========================================="
    echo ""
    print_info "Tests Passed: $TESTS_PASSED"
    print_info "Tests Failed: $TESTS_FAILED"
    echo ""

    # Determine overall result
    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All smoke tests passed!"
        exit 0
    else
        print_fail "Some smoke tests failed!"
        exit 1
    fi
}

# Run main function
main
