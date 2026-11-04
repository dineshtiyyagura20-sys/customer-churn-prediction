import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.
    """

    df = df.copy()

    # Remove unnecessary customer ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        # Fill missing TotalCharges using MonthlyCharges
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["MonthlyCharges"]
        )

    # Convert target variable
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({
            "Yes": 1,
            "No": 0
        })

    return df


def preprocess_data(path: str) -> pd.DataFrame:
    """
    Load and clean the dataset.
    """
    df = load_data(path)
    df = clean_data(df)

    return df
if __name__ == "__main__":

    input_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "data/processed/churn_cleaned.csv"

    df = preprocess_data(input_path)

    df.to_csv(output_path, index=False)

    print("Preprocessing completed successfully.")
    print(f"Saved cleaned dataset to: {output_path}")
    print(f"Shape: {df.shape}")