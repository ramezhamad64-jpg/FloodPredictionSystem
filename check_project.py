import os
import sys

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = [
    os.path.join("models", "lightgbm_direct_recursive_regression_bundle.pkl"),
    os.path.join("data", "processed", "regression_dataset.csv"),
    os.path.join("data", "processed", "classification_dataset.csv"),
]


def fail(message):
    print(f"FAIL: {message}")
    return False


def main():
    ok = True

    for relative_path in REQUIRED_FILES:
        full_path = os.path.join(BASE_DIR, relative_path)
        if os.path.exists(full_path):
            print(f"OK: {relative_path} exists")
        else:
            ok = fail(f"{relative_path} is missing") and ok

    bundle_path = os.path.join(BASE_DIR, "models", "lightgbm_direct_recursive_regression_bundle.pkl")
    regression_path = os.path.join(BASE_DIR, "data", "processed", "regression_dataset.csv")
    classification_path = os.path.join(BASE_DIR, "data", "processed", "classification_dataset.csv")

    if not os.path.exists(bundle_path) or not os.path.exists(regression_path):
        return 1

    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        fail(f"Could not load LightGBM bundle: {exc}")
        return 1

    if "features" in bundle:
        print(f"OK: bundle contains features ({len(bundle['features'])})")
    else:
        ok = fail('bundle is missing "features"') and ok

    if "direct_models" in bundle:
        print(f"OK: bundle contains direct_models ({len(bundle['direct_models'])})")
    else:
        ok = fail('bundle is missing "direct_models"') and ok

    if "features" not in bundle:
        return 1

    try:
        regression_columns = pd.read_csv(regression_path, nrows=0).columns
    except Exception as exc:
        fail(f"Could not read regression dataset columns: {exc}")
        return 1

    missing_features = [feature for feature in bundle["features"] if feature not in regression_columns]
    if missing_features:
        ok = fail("regression dataset is missing bundle features: " + ", ".join(missing_features)) and ok
    else:
        print("OK: regression dataset contains all bundle features")

    try:
        classification_columns = pd.read_csv(classification_path, nrows=0).columns
    except Exception as exc:
        fail(f"Could not read classification dataset columns: {exc}")
        return 1

    if "is_flood" in classification_columns:
        print('OK: classification dataset contains "is_flood"')
    else:
        ok = fail('classification dataset is missing "is_flood"') and ok

    if ok:
        print("Project check passed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
