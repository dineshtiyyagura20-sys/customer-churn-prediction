import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


DATA_PATH = "data/processed/churn_features.csv"
MODEL_PATH = "artifacts/model/model.joblib"
REPORT_DIR = "artifacts/model"


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # -----------------------------
    # 2. Separate features/target
    # -----------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------
    # 3. Same train/test split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Test data shape: {X_test.shape}")

    # -----------------------------
    # 4. Load trained model
    # -----------------------------

    model = joblib.load(MODEL_PATH)

    print("\nModel loaded successfully.")
    print(f"Model path: {MODEL_PATH}")

    # -----------------------------
    # 5. Predictions
    # -----------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # 6. Calculate metrics
    # -----------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # -----------------------------
    # 7. Print metrics
    # -----------------------------

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # -----------------------------
    # 8. Classification report
    # -----------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    # -----------------------------
    # 9. Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    # -----------------------------
    # 10. Save metrics
    # -----------------------------

    metrics = pd.DataFrame({
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ],
        "value": [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    })

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    metrics_path = os.path.join(
        REPORT_DIR,
        "evaluation_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nEvaluation metrics saved to: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()