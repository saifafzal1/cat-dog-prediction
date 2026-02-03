"""
Configuration loader utility module.

This module provides functions to load and validate configuration
from YAML files for the MLOps pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file. If None, uses the default
                    config file at configs/config.yaml

    Returns:
        Dictionary containing the configuration parameters

    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is malformed
    """
    # Determine the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Use default config path if not specified
    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    # Check if config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load and parse the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data-related configuration parameters.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary containing data configuration parameters
    """
    return config.get("data", {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model-related configuration parameters.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary containing model configuration parameters
    """
    return config.get("model", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training-related configuration parameters.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary containing training configuration parameters
    """
    return config.get("training", {})


def get_mlflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract MLflow-related configuration parameters.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary containing MLflow configuration parameters
    """
    return config.get("mlflow", {})
