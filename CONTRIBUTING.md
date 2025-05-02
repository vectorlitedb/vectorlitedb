# Contributing to VectorLiteDB

Thank you for your interest in contributing to VectorLiteDB! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Bug reports help us improve VectorLiteDB. When you submit a bug report, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Environment details (OS, Python version, etc.)
5. Any relevant logs or error messages

Please use the bug report template when filing an issue.

### Suggesting Features

We welcome feature suggestions! When suggesting a feature:

1. Provide a clear, descriptive title
2. Explain why this feature would be useful
3. Provide examples of how the feature would work
4. Describe any alternatives you've considered

Please use the feature request template when filing an issue.

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Add or update tests as necessary
5. Update documentation as needed
6. Submit a pull request

For significant changes, please open an issue first to discuss the proposed changes.

## Development Environment Setup

1. Clone your fork of the repository
   ```bash
   git clone https://github.com/vectorlitedb/vectorlitedb.git
   cd vectorlitedb
   ```

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks
   ```bash
   pre-commit install
   ```

## Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style with a few modifications:

- Line length: 88 characters (consistent with Black formatter)
- Use type hints for function signatures
- Use docstrings for all public methods, functions, and classes

We use the following tools to maintain code quality:
- [Black](https://black.readthedocs.io/en/stable/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [pytest](https://docs.pytest.org/en/stable/) for testing

## Testing

All new features and bug fixes should include tests. We use pytest for testing.

To run tests:
```bash
pytest
```

To run tests with coverage:
```bash
pytest --cov=vectorlitedb
```

## Documentation

Good documentation is crucial for VectorLiteDB. When contributing:

- Update or add docstrings for new functions, methods, and classes
- Update README.md if needed
- Add examples for new features
- Follow the Google style for docstrings

## Commit Messages

Write clear, concise commit messages that explain the "why" not just the "what":

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Fix bug" not "Fixes bug")
- Reference issues and pull requests when relevant

## Release Process

VectorLiteDB follows [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## Getting Help

If you need help with contributing:

- Ask questions in GitHub Discussions
- Reach out to maintainers

## Thank You!

Thank you for contributing to VectorLiteDB! Your efforts help improve the project for everyone.