# Contributing to Axion

Thank you for your interest in contributing to Axion! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/ax-foundry/axion.git
cd axion
```

2. Create a virtual environment (Python 3.12+ required):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test.py

# Run with coverage
coverage run -m pytest
coverage report
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

Key style guidelines:

- **Line length**: 88 characters
- **Quotes**: Single quotes for strings, double quotes for docstrings
- **Python version**: 3.12+

Run linting:

```bash
pre-commit run --all-files
```

## Optional Dependencies

Axion uses optional dependencies to keep the core lightweight. When adding features that require new dependencies, consider whether they should be optional:

```toml
# In pyproject.toml
[project.optional-dependencies]
your-feature = ["dependency>=1.0.0"]
```

Then add an import guard in your code:

```python
try:
    from some_package import SomeClass
except ImportError:
    raise ImportError(
        'Feature dependencies not installed. '
        'Install with: pip install axion[your-feature]'
    )
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes following the code style guidelines
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Ensure linting passes: `pre-commit run --all-files`
6. Update documentation if needed
7. Submit a pull request with a clear description of changes

## Questions?

Feel free to open an issue for questions or discussions.
