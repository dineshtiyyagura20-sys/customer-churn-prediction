import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for machine learning.
    """

    df = df.copy()

    # Separate target variable
    if "Churn" in df.columns:
        y = df["Churn"]
        X = df.drop(columns=["Churn"])
    else:
        X = df

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Convert categorical variables into dummy variables
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    # Combine target back with features
    if "Churn" in df.columns:
        X["Churn"] = y.values

    return X


def create_features(
    input_path: str,
    output_path: str
) -> pd.DataFrame:

    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Build features
    df_features = build_features(df)

    # Save feature dataset
    df_features.to_csv(
        output_path,
        index=False
    )

    print("Feature engineering completed successfully.")
    print(f"Saved feature dataset to: {output_path}")
    print(f"Shape: {df_features.shape}")

    return df_features


if __name__ == "__main__":

    input_path = "data/processed/churn_cleaned.csv"
    output_path = "data/processed/churn_features.csv"

    create_features(
        input_path,
        output_path
    )