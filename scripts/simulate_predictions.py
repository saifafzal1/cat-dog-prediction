#!/usr/bin/env python3
"""
Script to simulate predictions and test monitoring functionality.

This script sends simulated requests to the API to generate
metrics and test the monitoring endpoints.
"""

import os
import sys
import time
import random
import argparse
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_image(color: tuple = None) -> bytes:
    """
    Create a test image for prediction requests.

    Args:
        color: RGB color tuple, random if not specified

    Returns:
        Image bytes in JPEG format
    """
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    img = Image.new("RGB", (224, 224), color=color)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


def simulate_predictions(
    base_url: str, num_requests: int, delay: float = 0.5, add_labels: bool = False
) -> dict:
    """
    Simulate prediction requests to the API.

    Args:
        base_url: API base URL
        num_requests: Number of requests to send
        delay: Delay between requests in seconds
        add_labels: Whether to add true labels after predictions

    Returns:
        Dictionary with simulation statistics
    """
    stats = {
        "total": num_requests,
        "successful": 0,
        "failed": 0,
        "predictions": {"cat": 0, "dog": 0},
        "latencies": [],
    }

    print(f"Simulating {num_requests} prediction requests to {base_url}")
    print("-" * 50)

    for i in range(num_requests):
        try:
            # Create test image
            image_bytes = create_test_image()

            # Make prediction request
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
                timeout=30,
            )
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = result["confidence"]

                stats["successful"] += 1
                stats["predictions"][prediction] += 1
                stats["latencies"].append(latency)

                print(
                    f"[{i+1}/{num_requests}] Prediction: {prediction} "
                    f"(confidence: {confidence:.4f}, latency: {latency:.2f}ms)"
                )

                # Optionally add true label (simulated)
                if add_labels:
                    # Simulate true label (randomly correct 80% of the time)
                    if random.random() < 0.8:
                        true_label = prediction
                    else:
                        true_label = "dog" if prediction == "cat" else "cat"

                    # Add label to the prediction
                    label_response = requests.post(
                        f"{base_url}/predictions/-1/label", params={"true_label": true_label}
                    )
                    if label_response.status_code == 200:
                        print(f"         Added label: {true_label}")

            else:
                stats["failed"] += 1
                print(f"[{i+1}/{num_requests}] Failed: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            stats["failed"] += 1
            print(f"[{i+1}/{num_requests}] Error: {e}")

        # Delay between requests
        if delay > 0 and i < num_requests - 1:
            time.sleep(delay)

    print("-" * 50)
    print("\nSimulation Complete!")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Predictions: {stats['predictions']}")

    if stats["latencies"]:
        avg_latency = sum(stats["latencies"]) / len(stats["latencies"])
        print(f"  Avg Latency: {avg_latency:.2f}ms")

    return stats


def check_metrics(base_url: str) -> None:
    """
    Fetch and display current metrics.

    Args:
        base_url: API base URL
    """
    print("\n" + "=" * 50)
    print("Current Metrics")
    print("=" * 50)

    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            import json

            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed to fetch metrics: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def check_performance(base_url: str) -> None:
    """
    Fetch and display model performance metrics.

    Args:
        base_url: API base URL
    """
    print("\n" + "=" * 50)
    print("Model Performance")
    print("=" * 50)

    try:
        response = requests.get(f"{base_url}/performance")
        if response.status_code == 200:
            import json

            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Failed to fetch performance: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate predictions to test monitoring functionality"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to simulate (default: 10)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--add-labels", action="store_true", help="Add simulated true labels to predictions"
    )
    parser.add_argument(
        "--metrics-only", action="store_true", help="Only display current metrics, do not simulate"
    )

    args = parser.parse_args()

    # Check if API is available
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"API health check failed: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to API at {args.url}: {e}")
        sys.exit(1)

    if args.metrics_only:
        check_metrics(args.url)
        check_performance(args.url)
    else:
        # Run simulation
        simulate_predictions(
            base_url=args.url,
            num_requests=args.num_requests,
            delay=args.delay,
            add_labels=args.add_labels,
        )

        # Display metrics after simulation
        check_metrics(args.url)
        check_performance(args.url)


if __name__ == "__main__":
    main()
