import pandas as pd


def test_preprocessing_basic():
    data = pd.DataFrame({
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.0, 60.0, 70.0]
    })

    assert not data.empty
    assert "tenure" in data.columns
    assert "MonthlyCharges" in data.columns
    assert data["tenure"].dtype in ["int64", "int32"]