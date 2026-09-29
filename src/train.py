import sys, json, os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from data_loader import load_dataset, REQUIRED_FEATURES, TARGET
from pathlib import Path
# ...

RANDOM_STATE = 42
MARGIN = 0.10
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    df = load_dataset()
    X = df[REQUIRED_FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Baseline
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_score = f1_score(y_val, baseline.predict(X_val), zero_division=0)

    # Candidate
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])
    candidate.fit(X_train, y_train)
    candidate_score = f1_score(y_val, candidate.predict(X_val), zero_division=0)

    gate = baseline_score + MARGIN
    passed = candidate_score >= gate

    metrics = {
        "baseline_f1": round(float(baseline_score), 4),
        "candidate_f1": round(float(candidate_score), 4),
        "margin": MARGIN,
        "gate_threshold": round(float(gate), 4),
        "gate_passed": bool(passed),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }
    print(json.dumps(metrics, indent=2))

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if not passed:
        print(f"FATAL: quality gate failed "
              f"(candidate={candidate_score:.4f} < gate={gate:.4f})")
        sys.exit(1)

    joblib.dump(candidate, MODEL_DIR / "model.joblib")
    print("Model saved to", MODEL_DIR / "model.joblib")


if __name__ == "__main__":
    main()