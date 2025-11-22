# Contributing to RetentionAI

Thank you for your interest in contributing to RetentionAI! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Submitting Changes](#submitting-changes)
7. [ML/Data Science Guidelines](#mldata-science-guidelines)
8. [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- Basic understanding of machine learning concepts

### Local Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/RetentionAI.git
   cd RetentionAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up the database**
   ```bash
   python src/database.py
   ```

5. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Workflow Steps

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test types
   pytest tests/unit/
   pytest tests/integration/
   
   # Check code quality
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions or modifications
- `chore:` - Maintenance tasks

**Examples:**
```
feat(models): add XGBoost classifier support
fix(api): resolve authentication token validation
docs(readme): update installation instructions
```

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints for all function signatures

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Documentation Strings

All functions and classes must have docstrings:

```python
def train_model(data: pd.DataFrame, config: dict) -> Model:
    """
    Train a machine learning model on the provided data.
    
    Args:
        data: Training dataset with features and target
        config: Model configuration parameters
        
    Returns:
        Trained model instance
        
    Raises:
        ValueError: If data is empty or invalid
    """
    pass
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
├── e2e/           # End-to-end tests
└── fixtures/      # Test data and fixtures
```

### Writing Tests

1. **Unit Tests**
   - Test individual functions/methods
   - Mock external dependencies
   - Aim for 90%+ code coverage

2. **Integration Tests**
   - Test component interactions
   - Use test databases
   - Validate data flow

3. **ML Model Tests**
   - Test model training and prediction
   - Validate performance metrics
   - Test data preprocessing

### Test Example

```python
import pytest
from src.models.base import BaseModel

class TestBaseModel:
    def test_model_initialization(self):
        model = BaseModel(config={'param': 'value'})
        assert model.config['param'] == 'value'
    
    def test_model_training(self, sample_data):
        model = BaseModel()
        model.fit(sample_data)
        assert model.is_fitted
```

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description is clear and detailed

### Pull Request Process

1. **Fill out the PR template** with all required information
2. **Request appropriate reviewers** based on the changed components
3. **Respond to feedback** and make requested changes
4. **Ensure CI/CD checks pass** before merging

## ML/Data Science Guidelines

### Model Development

1. **Experiment Tracking**
   - Use MLflow for all experiments
   - Log parameters, metrics, and artifacts
   - Document model assumptions

2. **Data Validation**
   - Validate data quality before training
   - Check for data drift
   - Document data preprocessing steps

3. **Model Evaluation**
   - Use appropriate metrics for the problem
   - Test on holdout datasets
   - Evaluate model fairness and bias

4. **Reproducibility**
   - Set random seeds
   - Version datasets
   - Document environment requirements

### Model Deployment

1. **Model Validation**
   - Performance thresholds must be met
   - A/B testing when possible
   - Monitoring setup

2. **Documentation**
   - Model cards describing capabilities and limitations
   - Performance benchmarks
   - Known biases and mitigation strategies

## Documentation

### Types of Documentation

1. **Code Documentation**
   - Inline comments for complex logic
   - Docstrings for all public functions
   - Type hints

2. **User Documentation**
   - README updates
   - API documentation
   - Deployment guides

3. **Developer Documentation**
   - Architecture decisions
   - Setup instructions
   - Contributing guidelines

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date
- Use markdown for formatting

## Getting Help

### Resources

- **Documentation**: Check the README and docs/ folder
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions

### Contact

- **Maintainer**: @Saksham932007
- **Email**: [Project Email]
- **Discord/Slack**: [Community Chat]

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor statistics

Thank you for contributing to RetentionAI! 🚀