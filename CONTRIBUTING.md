# Contributing to RetentionAI

First off, thank you for considering contributing to RetentionAI! 🎉 It's people like you that make RetentionAI such a great tool for customer churn prediction.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How Can I Contribute?](#how-can-i-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Submitting Changes](#submitting-changes)
8. [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@retentionai.com](mailto:conduct@retentionai.com).

### Our Standards

**Examples of behavior that contributes to creating a positive environment include:**

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

**Examples of unacceptable behavior include:**

* The use of sexualized language or imagery and unwelcome sexual attention
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information without explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- **Python 3.10+** installed
- **Git** for version control
- **Docker** for containerized development (optional but recommended)
- Basic knowledge of machine learning concepts
- Familiarity with the project structure

### Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/RetentionAI.git
   cd RetentionAI
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Saksham932007/RetentionAI.git
   ```
4. **Create a development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/Saksham932007/RetentionAI/issues) to avoid duplicates.

**When filing a bug report, please include:**

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, browser if applicable)
- **Screenshots** (if applicable)
- **Error logs** or stack traces

**Bug Report Template:**
```markdown
## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. Ubuntu 22.04, Windows 11, macOS 13]
- Python Version: [e.g. 3.10.6]
- RetentionAI Version: [e.g. 1.2.0]
- Browser: [e.g. Chrome 118, Firefox 119] (if applicable)

## Additional Context
Add any other context about the problem here.
```

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear title and description**
- **Detailed explanation** of the proposed functionality
- **Use cases** and benefits
- **Potential implementation** approach (if you have ideas)
- **Mockups or diagrams** (if applicable)

### 📝 Contributing Documentation

Documentation contributions are highly valuable:

- **API documentation** improvements
- **Tutorial** enhancements
- **Example** additions
- **Translation** to other languages
- **FAQ** updates

### 🔧 Code Contributions

We welcome code contributions in these areas:

#### Machine Learning
- **New algorithms** or model improvements
- **Feature engineering** enhancements
- **Hyperparameter optimization** improvements
- **Model interpretability** features

#### Web Application
- **UI/UX improvements**
- **New dashboard features**
- **Performance optimizations**
- **Accessibility enhancements**

#### Infrastructure
- **Containerization** improvements
- **CI/CD** pipeline enhancements
- **Monitoring** additions
- **Security** improvements

#### Data Processing
- **ETL pipeline** optimizations
- **Data validation** enhancements
- **Feature store** improvements
- **Database** optimizations

## Development Setup

### Local Development Environment

1. **Set up the environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install
   ```

2. **Set up the database**:
   ```bash
   python src/database.py
   ```

3. **Run the application**:
   ```bash
   streamlit run src/app.py
   ```

### Docker Development Environment

1. **Build and run with Docker Compose**:
   ```bash
   # Development environment
   docker-compose -f docker-compose.dev.yml up --build

   # Production-like environment
   docker-compose up --build
   ```

2. **Run tests in Docker**:
   ```bash
   docker-compose exec app pytest tests/
   ```

### Development Tools

We recommend using these tools for development:

- **IDE**: VS Code with Python and Docker extensions
- **Code Formatting**: Black, isort
- **Linting**: Ruff, mypy
- **Testing**: pytest, coverage
- **Pre-commit**: Automated code quality checks

### Project Structure

Understanding the project structure will help you contribute effectively:

```
RetentionAI/
├── src/                    # Application source code
│   ├── app.py             # Main Streamlit application
│   ├── model_trainer.py   # ML training pipeline
│   ├── data_processor.py  # Data processing
│   ├── feature_engineer.py # Feature engineering
│   ├── monitoring.py      # Metrics and monitoring
│   └── health_endpoints.py # Health check API
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/             # End-to-end tests
├── data/                 # Data storage
├── models/               # Model artifacts
├── config/               # Configuration files
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── .github/              # CI/CD workflows
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes preferred
- **Import ordering**: isort configuration

### Code Formatting

Use these tools for consistent formatting:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings
- **Type hints**: Required for all functions
- **Comments**: Explain why, not what
- **README updates**: Update documentation for new features

**Example function with proper documentation:**

```python
def predict_churn(
    customer_data: Dict[str, Any],
    model_version: Optional[str] = None
) -> Dict[str, Union[float, str, List[str]]]:
    """Predict customer churn probability.

    Args:
        customer_data: Dictionary containing customer features.
        model_version: Specific model version to use. If None, uses
            the latest model.

    Returns:
        Dictionary containing:
            - churn_probability: Float between 0 and 1
            - prediction: "Yes" or "No"
            - confidence: "High", "Medium", or "Low"
            - recommendations: List of recommended actions

    Raises:
        ValueError: If customer_data is missing required features.
        ModelNotFoundError: If specified model_version doesn't exist.

    Example:
        >>> customer = {"tenure": 12, "monthly_charges": 50.0, ...}
        >>> result = predict_churn(customer)
        >>> print(f"Churn probability: {result['churn_probability']:.2%}")
    """
    # Implementation here
    pass
```

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

**Examples:**
```
feat(ml): Add ensemble model support

Add support for ensemble models combining XGBoost, CatBoost, and LightGBM.
Includes automatic model weighting and cross-validation.

Closes #123

fix(api): Handle missing customer features gracefully

Previously would throw KeyError when required features were missing.
Now returns validation error with clear message.

docs(readme): Update installation instructions

Add Docker installation option and troubleshooting section.
```

## Testing Guidelines

### Test Structure

We use pytest for testing with this structure:

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_model.py
│   ├── test_features.py
│   └── test_utils.py
├── integration/           # Integration tests (slower, components)
│   ├── test_pipeline.py
│   ├── test_database.py
│   └── test_api.py
└── e2e/                  # End-to-end tests (slowest, full system)
    ├── test_prediction_flow.py
    └── test_user_workflows.py
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from src.feature_engineer import FeatureEngineer

class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        self.sample_data = {
            "tenure": 12,
            "monthly_charges": 50.0,
            "contract": "Month-to-month"
        }

    def test_tenure_buckets(self):
        """Test tenure bucketing functionality."""
        result = self.engineer.create_tenure_buckets(self.sample_data)
        assert result["tenure_bucket"] == "Medium"

    def test_missing_features_handling(self):
        """Test handling of missing features."""
        incomplete_data = {"tenure": 12}  # Missing required features
        
        with pytest.raises(ValueError, match="Missing required features"):
            self.engineer.engineer_features(incomplete_data)

    @pytest.mark.parametrize("tenure,expected_bucket", [
        (1, "New"),
        (12, "Medium"),
        (48, "Long"),
        (72, "Veteran")
    ])
    def test_tenure_bucket_boundaries(self, tenure, expected_bucket):
        """Test tenure bucket boundary conditions."""
        data = {**self.sample_data, "tenure": tenure}
        result = self.engineer.create_tenure_buckets(data)
        assert result["tenure_bucket"] == expected_bucket
```

### Test Coverage

Maintain test coverage above 90%:

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Performance Testing

Include performance tests for critical paths:

```python
import time
import pytest

class TestPerformance:
    """Performance test suite."""

    def test_prediction_latency(self):
        """Test that predictions complete within acceptable time."""
        start_time = time.time()
        
        # Run prediction
        result = predict_churn(self.sample_customer)
        
        elapsed_time = time.time() - start_time
        
        # Should complete within 100ms
        assert elapsed_time < 0.1
        assert result is not None
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the full test suite**:
   ```bash
   # Run all tests
   pytest tests/ -v

   # Run code quality checks
   pre-commit run --all-files

   # Check test coverage
   pytest tests/ --cov=src --cov-fail-under=90
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat(ml): Add amazing feature"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request** on GitHub

### Pull Request Guidelines

**Pull Request Template:**
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

**Review Criteria:**
- **Functionality**: Does the code do what it's supposed to do?
- **Code Quality**: Is the code clean, readable, and maintainable?
- **Tests**: Are there adequate tests with good coverage?
- **Documentation**: Is the documentation updated and clear?
- **Performance**: Does the change impact performance negatively?
- **Security**: Are there any security implications?

### Continuous Integration

Our CI pipeline runs on every pull request:

1. **Linting and Formatting**: Ruff, Black, isort
2. **Type Checking**: mypy
3. **Unit Tests**: pytest with coverage reporting
4. **Integration Tests**: Database and API tests
5. **Security Scanning**: Bandit and safety checks
6. **Docker Build**: Ensure containers build successfully
7. **Documentation**: Check for broken links and formatting

All checks must pass before merging.

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples: `1.0.0`, `1.1.0`, `1.1.1`

### Release Workflow

1. **Feature Freeze**: Stop adding new features
2. **Release Candidate**: Create RC for testing
3. **Testing**: Comprehensive testing in staging
4. **Documentation**: Update all documentation
5. **Release**: Tag and deploy to production
6. **Post-Release**: Monitor and address issues

### Maintainer Guidelines

For project maintainers:

```bash
# Create release branch
git checkout -b release/v1.2.0

# Update version numbers
# Update CHANGELOG.md
# Final testing

# Merge to main
git checkout main
git merge release/v1.2.0

# Tag release
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0

# Deploy to production
./scripts/deploy.sh production
```

## Getting Help

### Communication Channels

- **GitHub Discussions**: General questions and feature discussions
- **GitHub Issues**: Bug reports and specific feature requests
- **Slack**: Real-time communication (link in README)
- **Email**: maintainers@retentionai.com

### Mentorship

New contributors can request mentorship:

- **First-time contributors**: We'll help you find good first issues
- **Pair programming**: Available for complex features
- **Code reviews**: Detailed feedback and learning opportunities
- **Architecture discussions**: Understanding design decisions

### Good First Issues

Look for issues labeled `good first issue`:

- Documentation improvements
- Simple bug fixes
- Test additions
- Configuration updates

### Recognition

Contributors are recognized through:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Mentioned in changelog
- **Community highlights**: Featured in newsletters
- **Swag**: Stickers and t-shirts for significant contributors

## Thank You! 🙏

Every contribution matters, whether it's:

- Reporting a bug
- Suggesting a feature
- Writing documentation
- Submitting code
- Helping other users

Thank you for helping make RetentionAI better for everyone!

---

*This contributing guide is a living document. Please suggest improvements through issues or pull requests.*