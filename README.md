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
