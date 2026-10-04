import argparse
import json
from pathlib import Path
import pandas as pd
from churn_model import ChurnPredictor

def parse_args():

    parser = argparse.ArgumentParser(description="Train the churn model.")

    parser.add_argument("--data", default="telco_customer_churn.csv")
    parser.add_argument("--model-out", default="churn_model.pkl")
    parser.add_argument("--output-dir", default="artifacts")

    return parser.parse_args()

def main():

    args = parse_args()

    df = pd.read_csv(args.data)

    print(f"Dataset shape: {df.shape}")

    if "Churn" not in df.columns:

        raise ValueError(
            "Target column 'Churn' was not found."
        )

    duplicates = int(df.duplicated().sum())

    if duplicates:

        print(f"WARNING: {duplicates} duplicated rows found.")

    y = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    if y.isna().any():

        raise ValueError("Unexpected values found in Churn column.")

    y = y.astype(int)

    predictor = ChurnPredictor()

    missing_features = [
        feature
        for feature in predictor.MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    X = df[
        predictor.MODEL_FEATURES
    ].copy()

    print(f"\nFeatures used: {len(X.columns)}")
    print("\nTarget distribution:")
    print(y.value_counts())

    X_test, y_test = predictor.train(X, y)
    metrics = predictor.evaluate(X_test, y_test)

    print("MODEL EVALUATION RESULTS")
    
    print(f"Test ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"Test PR-AUC   : {metrics['pr_auc']:.3f}")
    print(f"Test Precision: {metrics['precision']:.3f}")
    print(f"Test Recall   : {metrics['recall']:.3f}")

    cv = metrics["cv"]
    print(f"\n{cv['folds']}-Fold Cross-Validation (Training Set):")
    print(f"CV ROC-AUC    : {cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f}")
    print(f"CV PR-AUC     : {cv['pr_auc_mean']:.3f} ± {cv['pr_auc_std']:.3f}")

    print("\nConfusion Matrix (Threshold = 0.50):")
    for row in metrics["confusion_matrix"]:
        print(row)

    predictor.save_model(args.model_out)

    # Reporting artifacts (README tables, later analysis)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_prob": predictor.predict_proba(X_test)
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nModel artifact: {args.model_out}")
    print(f"Reports written to: {output_dir}/")

if __name__ == "__main__":
    main()