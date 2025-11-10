# TestEnv Docker Environment

This Docker setup replicates the `testenv` Python 3.12.0 virtual environment for running pytest tests in a containerized environment. It ensures consistent test execution across different systems.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and run all tests
docker-compose -f docker-compose.testenv.yml up --build testenv

# Run specific test categories
docker-compose -f docker-compose.testenv.yml up --build testenv-unit
docker-compose -f docker-compose.testenv.yml up --build testenv-integration
docker-compose -f docker-compose.testenv.yml up --build testenv-performance
```

### Option 2: Docker Build and Run

```bash
# Build the Docker image
docker build -f Dockerfile.testenv -t data-pipeline-testenv .

# Run all tests
docker run --rm -v "$(pwd)/tests/logs:/app/Data-Pipeline/tests/logs" data-pipeline-testenv

# Run with interactive shell for debugging
docker run --rm -it -v "$(pwd):/app/Data-Pipeline" data-pipeline-testenv /bin/bash

# Run specific test file
docker run --rm data-pipeline-testenv /opt/testenv/bin/python -m pytest tests/test_preprocessing.py -v

# Run with coverage report
docker run --rm -v "$(pwd)/tests/logs:/app/Data-Pipeline/tests/logs" data-pipeline-testenv \
  /opt/testenv/bin/python -m pytest tests/ -v --cov=scripts --cov-report=html:tests/logs/coverage_html
```

## Environment Details

### Python Environment
- **Python Version**: 3.12.0 (matches local testenv)
- **Virtual Environment**: `/opt/testenv` (replicates local testenv)
- **Package Manager**: pip (latest)

### Key Dependencies
- pytest 7.0.0+ (testing framework)
- torch 1.11.0+ (ML framework)
- transformers 4.20.0+ (NLP models)
- pandas 1.5.0+ (data processing)
- All dependencies from `requirements.txt`

### Test Categories
- **Unit Tests**: `tests/unit/` - Individual component testing
- **Integration Tests**: `tests/integration/` - End-to-end pipeline testing
- **Performance Tests**: `tests/performance/` - Benchmarking and performance validation
- **Bias Tests**: `tests/bias/` - Bias detection and analysis
- **Anomaly Tests**: `tests/anomaly/` - Anomaly detection testing

## File Structure

```
Data-Pipeline/
├── Dockerfile.testenv              # Main Docker configuration
├── docker-compose.testenv.yml      # Docker Compose setup
├── .dockerignore                   # Docker build optimization
├── requirements.txt                # Python dependencies
├── tests/
│   ├── logs/                      # Test output and reports (mounted)
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── performance/               # Performance tests
│   └── ...
└── scripts/                       # Source code to test
```

## Usage Examples

### Development Workflow

```bash
# 1. Build the environment
docker-compose -f docker-compose.testenv.yml build

# 2. Run all tests with coverage
docker-compose -f docker-compose.testenv.yml up testenv

# 3. Check results
cat tests/logs/junit-results.xml           # JUnit XML results
open tests/logs/coverage_html/index.html   # Coverage report
```

### Continuous Integration

```yaml
# Example GitHub Actions workflow
- name: Run Tests in Docker
  run: |
    docker-compose -f docker-compose.testenv.yml up --build --exit-code-from testenv testenv
```

### Debugging Failed Tests

```bash
# Run with interactive shell
docker run --rm -it -v "$(pwd):/app/Data-Pipeline" data-pipeline-testenv /bin/bash

# Inside container, run specific tests
/opt/testenv/bin/python -m pytest tests/test_specific.py::TestClass::test_method -v -s
```

## Output and Reports

### Test Results
- **Console Output**: Real-time test execution status
- **JUnit XML**: `tests/logs/junit-results.xml` (for CI integration)
- **Coverage Report**: `tests/logs/coverage_html/` (HTML format)

### Log Files
- Test execution logs are saved to `tests/logs/`
- Mount the logs directory to persist results: `-v "$(pwd)/tests/logs:/app/Data-Pipeline/tests/logs"`

## Performance Expectations

Based on the current test suite:
- **Total Tests**: 119 tests
- **Expected Success Rate**: 100% (all tests should pass)
- **Execution Time**: ~30-60 seconds (depending on system)
- **Memory Usage**: ~500MB peak during ML model loading

## Troubleshooting

### Common Issues

1. **Memory Issues**: Increase Docker memory limit to 4GB+
   ```bash
   docker run --memory=4g data-pipeline-testenv
   ```

2. **Permission Issues**: Ensure proper volume mounting
   ```bash
   docker run --rm -v "$(pwd):/app/Data-Pipeline:rw" data-pipeline-testenv
   ```

3. **Missing Dependencies**: Rebuild the image
   ```bash
   docker-compose -f docker-compose.testenv.yml build --no-cache
   ```

### Verification Commands

```bash
# Check Python version
docker run --rm data-pipeline-testenv /opt/testenv/bin/python --version

# Verify key packages
docker run --rm data-pipeline-testenv /opt/testenv/bin/python -c "import pytest, torch, transformers; print('All imports successful')"

# Test pytest installation
docker run --rm data-pipeline-testenv /opt/testenv/bin/python -m pytest --version
```

## Integration with CI/CD

This Docker setup is designed for easy integration with CI/CD pipelines:

- **GitHub Actions**: Use the docker-compose command in workflow
- **Jenkins**: Run as Docker container in pipeline
- **GitLab CI**: Use the Dockerfile in `.gitlab-ci.yml`
- **Local Development**: Consistent environment across team members

## Notes

- The container replicates the exact `testenv` setup from Windows Python 3.12.0
- All 119 tests should pass (achieving 100% success rate)
- ML model downloads are cached within the container for faster subsequent runs
- Test logs and reports are mounted to host for easy access