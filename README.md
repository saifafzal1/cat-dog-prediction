# MLOps Assignment 2 - Cats vs Dogs Classification

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs) for a pet adoption platform.

## Project Structure

```
.
├── configs/
│   └── config.yaml          # Configuration parameters
├── data/
│   ├── raw/                  # Raw dataset (DVC tracked)
│   └── processed/            # Preprocessed data (DVC tracked)
├── models/                   # Trained models (DVC tracked)
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   ├── download.py       # Dataset download utilities
│   │   ├── preprocess.py     # Data preprocessing
│   │   └── dataloader.py     # PyTorch data loaders
│   ├── models/
│   │   └── cnn.py            # CNN model architecture
│   ├── utils/
│   │   └── config.py         # Configuration utilities
│   └── train.py              # Training script with MLflow
├── tests/                    # Unit tests
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Kaggle API (for dataset download)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings and create a new API token
3. Place the downloaded `kaggle.json` in `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 4. Initialize Git and DVC

```bash
git init
dvc init
```

## Running the Pipeline

### Option 1: Run with DVC (Recommended)

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro download_data
dvc repro preprocess
dvc repro train
```

### Option 2: Run Scripts Individually

```bash
# Download dataset
python src/data/download.py

# Preprocess data
python src/data/preprocess.py

# Train model
python src/train.py
```

## Experiment Tracking with MLflow

View experiments in the MLflow UI:

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 in your browser.

## Configuration

Edit `configs/config.yaml` to modify:
- Data split ratios
- Image size
- Model parameters
- Training hyperparameters
- Augmentation settings

## Milestones

- **M1**: Model Development & Experiment Tracking (Current)
- **M2**: Model Packaging & Containerization
- **M3**: CI Pipeline for Build, Test & Image Creation
- **M4**: CD Pipeline & Deployment
- **M5**: Monitoring, Logs & Final Submission
