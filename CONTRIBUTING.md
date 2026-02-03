# Contributing Guide

This document describes the development workflow and CI/CD pipeline for the Cats vs Dogs Classification project.

## Development Setup

### Prerequisites

- Python 3.9+
- Docker
- Git

### Local Setup

```bash
# Clone the repository
git clone <repository-url>
cd "MLOPS Assignment 2"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the project style guide
- Add/update unit tests for new functionality
- Update documentation as needed

### 3. Run Local Checks

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make test-cov
```

### 4. Commit Changes

Pre-commit hooks will automatically run on commit:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Security checks (Bandit)

```bash
git add .
git commit -m "feat: description of changes"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Create a Pull Request on GitHub.

## CI/CD Pipeline

### Pipeline Overview

```
Push/PR -> Lint -> Test -> Build -> Security Scan -> Publish
```

### CI Jobs

| Job | Description | Trigger |
|-----|-------------|---------|
| **lint** | Code quality checks (Black, isort, Flake8) | All pushes and PRs |
| **test** | Unit tests with pytest | After lint passes |
| **build** | Build and push Docker image | After tests pass |
| **security-scan** | Trivy vulnerability scan | On main branch only |
| **publish** | Push to Docker Hub | On main branch only |

### GitHub Actions Workflows

1. **ci.yaml** - Main CI pipeline
   - Runs on push to main/develop and all PRs
   - Builds and pushes to GitHub Container Registry

2. **docker-publish.yaml** - Docker Hub publishing
   - Triggered after successful CI on main branch
   - Pushes to Docker Hub

3. **pr-validation.yaml** - PR validation
   - Quick validation for pull requests
   - Tests Docker build and container health

### Required Secrets

Configure these in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

GitHub Container Registry uses `GITHUB_TOKEN` automatically.

## Code Style

### Python Style Guide

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestSimpleCNN::test_forward_pass_shape -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use pytest fixtures for common setup

## Docker

### Building Locally

```bash
make docker-build
```

### Running Locally

```bash
make docker-run
```

### Testing Container

```bash
make docker-test
```

## Troubleshooting

### Common Issues

1. **Pre-commit hooks fail**: Run `make format` before committing
2. **Tests fail**: Check for missing dependencies with `make install`
3. **Docker build fails**: Ensure Docker daemon is running

### Getting Help

- Check existing issues on GitHub
- Create a new issue with detailed description
