# MLOps Lab

This project provides a minimal MLOps starter structure for training, validating, and predicting with a machine learning model.

## Project structure

- `src/data_loader.py` loads the data
- `src/train.py` trains the model
- `src/validate.py` validates model performance
- `src/predict.py` generates predictions
- `tests/test_app.py` checks the basic workflow

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Example run

```bash
python src/train.py
python src/validate.py
python src/predict.py
```

## CI/CD

The GitHub Actions workflow in `.github/workflows/pipeline.yml` runs lint/test steps on pushes and pull requests.
