#!/bin/bash
# Setup script for Python skeleton project
# Usage: bash setup-project.sh

set -e

echo "🚀 Setting up Python skeleton project..."
echo ""

# Create directory structure
echo "📁 Creating directories..."
mkdir -p src tests notebooks data .vscode

# Create __init__.py files
touch src/__init__.py tests/__init__.py

# Create src/main.py
cat > src/main.py << 'EOF'
"""
Main entry point for the project.
"""

import numpy as np
from src.utils import greet


def main():
    """Main function to run the application."""
    print(greet("World"))
    
    # Example: Use numpy for quick testing
    data = np.array([1, 2, 3, 4, 5])
    print(f"Data mean: {np.mean(data)}")
    print(f"Data std: {np.std(data)}")


if __name__ == "__main__":
    main()
EOF

# Create src/utils.py
cat > src/utils.py << 'EOF'
"""
Utility functions for the project.
"""


def greet(name: str) -> str:
    """
    Return a greeting message.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"


def add(a: float, b: float) -> float:
    """
    Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    return a * b
EOF

# Create tests/test_main.py
cat > tests/test_main.py << 'EOF'
"""
Tests for utility functions.
"""

import pytest
from src.utils import greet, add, multiply


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_greet(self):
        """Test the greet function."""
        assert greet("Alice") == "Hello, Alice!"
        assert greet("") == "Hello, !"
    
    def test_add(self):
        """Test the add function."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0.5, 0.5) == 1.0
    
    def test_multiply(self):
        """Test the multiply function."""
        assert multiply(2, 3) == 6
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0
    
    @pytest.mark.slow
    def test_add_with_large_numbers(self):
        """Test add with large numbers (marked as slow)."""
        assert add(1e10, 1e10) == 2e10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# Create tests/conftest.py
cat > tests/conftest.py << 'EOF'
"""
Pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_dict():
    """Provide sample dictionary for tests."""
    return {"a": 1, "b": 2, "c": 3}


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test (if needed)."""
    yield
    # Cleanup code here (if needed)
EOF

# Create .vscode/settings.json
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "pylint.enabled": true,
  "pylint.lintOnSave": true,
  "pylint.args": ["--disable=C0111"],
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "editor.rulers": [88, 120],
  "editor.wordWrap": "on",
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.venv": true
  },
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "terminal.integrated.defaultProfile.linux": "bash"
}
EOF

# Create .vscode/launch.json
cat > .vscode/launch.json << 'EOF'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Python: Main Entry Point",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/main.py",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "Python: Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
EOF

# Create .vscode/extensions.json
cat > .vscode/extensions.json << 'EOF'
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-kernel-launcher",
    "charliermarsh.ruff",
    "eamodio.gitlens"
  ]
}
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
scipy==1.11.2
EOF

# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
black==23.7.0
pylint==2.17.5
mypy==1.4.1
ruff==0.0.283
ipython==8.14.0
jupyter==1.0.0
EOF

# Create pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
.venv/
venv/
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.coverage
htmlcov/
.vscode/
.idea/
*.swp
.DS_Store
.ipynb_checkpoints
*.ipynb
.mypy_cache/
.env
data/
logs/
*.log
EOF

# Create README.md
cat > README.md << 'EOF'
# Project Name

A Python testing environment skeleton for iterative development before porting to Google Colab Pro.

## Quick Start

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run the Application
```bash
python src/main.py
```

### 4. Run Tests
```bash
pytest              # Run all tests
pytest -v           # Verbose output
pytest --cov=src    # With coverage report
```

## Project Structure
```
├── src/             # Source code
├── tests/           # Test files
├── notebooks/       # Jupyter notebooks
├── data/            # Data files
└── .vscode/         # VS Code config
```

## Debugging
1. Set breakpoints in VS Code
2. Press **F5** to debug
3. Use Debug Console for inspection

---
Created with ❤️ for efficient development workflows.
EOF

echo "✅ All files created!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Create virtual environment:"
echo "   python -m venv .venv"
echo ""
echo "2️⃣  Activate it:"
echo "   source .venv/bin/activate"
echo ""
echo "3️⃣  Install dependencies:"
echo "   pip install -r requirements.txt"
echo "   pip install -r requirements-dev.txt"
echo ""
echo "4️⃣  Open in VS Code:"
echo "   code ."
echo ""
echo "🎉 Done! You're ready to code."
