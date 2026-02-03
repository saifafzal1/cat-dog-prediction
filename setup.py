"""
Setup script for the MLOps Cats vs Dogs Classification project.

This allows the project to be installed as a package for easier imports.
"""

from setuptools import setup, find_packages

setup(
    name="cats_dogs_classifier",
    version="1.0.0",
    description="MLOps pipeline for Cats vs Dogs binary image classification",
    author="MLOps Student",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "mlflow>=2.9.0",
        "dvc>=3.30.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-model=src.train:main",
        ]
    },
)
