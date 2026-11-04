import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel


# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


from src.models.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0"
)


class Customer(BaseModel):

    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):

    prediction, probability = predict_churn(
        customer.model_dump()
    )

    if prediction == 1:
        result = "CUSTOMER IS LIKELY TO CHURN"
    else:
        result = "CUSTOMER IS LIKELY TO STAY"

    return {
        "prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "result": result
    }