import sys
import pandas as pd

from data_loader import load_dataset, REQUIRED_FEATURES, TARGET


def validate(df: pd.DataFrame) -> None:
    errors = []

    # 1. Required feature columns present
    missing_features = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing_features:
        errors.append(f"Missing feature columns: {missing_features}")

    # 2. Target column present
    if TARGET not in df.columns:
        errors.append(f"Missing target column: {TARGET}")

    # 3. Enough rows
    if len(df) < 50:
        errors.append(f"Too few rows: {len(df)} (need >= 50)")

    # 4. Target has 2+ classes (classification)
    if TARGET in df.columns and df[TARGET].nunique() < 2:
        errors.append(f"Target has only {df[TARGET].nunique()} class(es)")

    # 5. No excessive missing values in features
    if all(c in df.columns for c in REQUIRED_FEATURES):
        null_frac = df[REQUIRED_FEATURES].isnull().mean()
        bad = null_frac[null_frac > 0.5].to_dict()
        if bad:
            errors.append(f"Columns with >50% nulls: {bad}")

    if errors:
        for e in errors:
            print(f"VALIDATION ERROR: {e}")
        sys.exit(1)

    print(f"Validation passed: {df.shape[0]} rows, "
          f"{len(REQUIRED_FEATURES)} features, "
          f"target classes = {sorted(df[TARGET].unique().tolist())}")


def main():
    df = load_dataset()
    validate(df)


if __name__ == "__main__":
    main()