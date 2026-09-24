# MLOps Lab — Wine Quality Classifier

## Setup
pip install -r requirements.txt
cd src && python train.py
cd .. && pytest tests/

## Dataset
- Source: UCI Wine Quality (Red) — URL in data_loader.py
- Target: `target` (1 if quality ≥ 7)
- Features: 11 chemical properties
- Task: Binary classification

## Metric: F1-score
Chosen because the target is imbalanced (~13% positive).
Accuracy would be misleading (a majority-class dummy scores ~87%).

## Margin: 0.10
DummyClassifier F1 ≈ 0. A margin of 0.10 ensures the model
captures real signal, not noise. Too low → broken model passes.
Too high → good model fails unnecessarily.

## Runs
- Failure A (quality gate): <URL>
- Failure B (app test): <URL>
- Final success: <URL>
- Artifact: `model-package-<run_number>`

## Answers
1. **Why F1?** Imbalanced classes; F1 balances precision/recall.
2. **Why 0.10 margin?** See above — floor for meaningful signal.
3. **Failure causes:** A = weak model below gate; B = predict.py
   no longer rejected missing features, so test_missing_feature_rejected
   failed. Both blocked the artifact upload step.
4. **CI/CD demonstrated by:** workflow on push to main, automated
   training/tests, gated artifact upload = continuous integration +
   artifact delivery.
5. **Maturity level: Level 1–2 (Manual/automated ML pipeline).**
   We have automated training + tests + artifact publishing, but no
   continuous training trigger, no model registry, no monitoring.
   Next level requires automated retraining triggers and a registry.