import os
import joblib
import pandas as pd


MODEL_PATH = "artifacts/model/model.joblib"
FEATURES_PATH = "data/processed/churn_features.csv"


def load_model():
    """Load trained model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_churn(sample_customer):
    """Predict customer churn."""

    # Load trained model
    model = load_model()

    # Load feature dataset
    feature_df = pd.read_csv(FEATURES_PATH)

    # Remove target column
    feature_columns = [
        column
        for column in feature_df.columns
        if column != "Churn"
    ]

    # Convert sample customer to DataFrame
    sample_df = pd.DataFrame([sample_customer])

    # Create dummy variables
    sample_df = pd.get_dummies(
        sample_df,
        drop_first=True,
        dtype=int
    )

    # Make sure sample has exactly the same
    # columns used during training
    sample_df = sample_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(sample_df)[0]

    probability = model.predict_proba(
        sample_df
    )[0][1]

    return prediction, probability


if __name__ == "__main__":

    # Example customer
    sample_customer = {

        "SeniorCitizen": 0,

        "tenure": 12,

        "MonthlyCharges": 70.50,

        "TotalCharges": 850.00,

        "gender": "Male",

        "Partner": "Yes",

        "Dependents": "No",

        "PhoneService": "Yes",

        "MultipleLines": "No phone service",

        "InternetService": "DSL",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "No",

        "StreamingMovies": "No",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check"
    }

    prediction, probability = predict_churn(
        sample_customer
    )

    print()
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    print(f"Churn Prediction : {prediction}")
    print(f"Churn Probability: {probability:.2%}")

    print()

    if prediction == 1:

        print(
            "Result: CUSTOMER IS LIKELY TO CHURN"
        )

    else:

        print(
            "Result: CUSTOMER IS LIKELY TO STAY"
        )

    print("=" * 60)