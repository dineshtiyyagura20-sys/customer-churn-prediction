from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 846.0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data
    assert "result" in data