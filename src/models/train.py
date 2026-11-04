import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


DATA_PATH = "data/processed/churn_features.csv"
MODEL_DIR = "artifacts/model"

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load feature-engineered dataset."""

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


def build_preprocessor(X):
    """Build preprocessing pipeline."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------

    df = load_data()

    # -----------------------------
    # 2. Separate features/target
    # -----------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("\nTarget distribution:")
    print(y.value_counts())

    # -----------------------------
    # 3. Train/Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print(f"Training data: {X_train.shape}")
    print(f"Testing data : {X_test.shape}")

    # -----------------------------
    # 4. Preprocessor
    # -----------------------------

    preprocessor = build_preprocessor(X_train)

    # -----------------------------
    # 5. Define models
    # -----------------------------

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=6,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
    }

    # -----------------------------
    # 6. Train models
    # -----------------------------

    results = {}

    trained_models = {}

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        print(f"\nTraining {name}...")

        pipeline.fit(
            X_train,
            y_train
        )

        result = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        results[name] = result
        trained_models[name] = pipeline

    # -----------------------------
    # 7. Compare models
    # -----------------------------

    results_df = pd.DataFrame(
        results.values()
    )

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.sort_values(
            by="f1",
            ascending=False
        ).to_string(index=False)
    )

    # -----------------------------
    # 8. Select best model
    # -----------------------------

    best_model_name = (
        results_df
        .sort_values(
            by="f1",
            ascending=False
        )
        .iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    print("\n")
    print("=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 70)

    # -----------------------------
    # 9. Save best model
    # -----------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(f"\nBest model saved to: {model_path}")

    # -----------------------------
    # 10. Save comparison results
    # -----------------------------

    results_path = os.path.join(
        MODEL_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"Model comparison saved to: {results_path}"
    )


if __name__ == "__main__":
    main()