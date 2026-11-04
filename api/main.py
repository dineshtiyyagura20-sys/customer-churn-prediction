import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel, Field, AliasChoices


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)


# ---------------------------------------------------------
# Import prediction function
# ---------------------------------------------------------

from src.models.predict import predict_churn


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0"
)


# ---------------------------------------------------------
# Customer input schema
# ---------------------------------------------------------

class Customer(BaseModel):

    # Numerical features

    SeniorCitizen: int = Field(
        validation_alias=AliasChoices(
            "senior_citizen",
            "SeniorCitizen"
        )
    )

    tenure: int

    MonthlyCharges: float = Field(
        validation_alias=AliasChoices(
            "monthly_charges",
            "MonthlyCharges"
        )
    )

    TotalCharges: float = Field(
        validation_alias=AliasChoices(
            "total_charges",
            "TotalCharges"
        )
    )

    # Customer information

    gender: str

    Partner: str = Field(
        validation_alias=AliasChoices(
            "partner",
            "Partner"
        )
    )

    Dependents: str = Field(
        validation_alias=AliasChoices(
            "dependents",
            "Dependents"
        )
    )

    # Phone and internet services

    PhoneService: str = Field(
        validation_alias=AliasChoices(
            "phone_service",
            "PhoneService"
        )
    )

    MultipleLines: str = Field(
        validation_alias=AliasChoices(
            "multiple_lines",
            "MultipleLines"
        )
    )

    InternetService: str = Field(
        validation_alias=AliasChoices(
            "internet_service",
            "InternetService"
        )
    )

    OnlineSecurity: str = Field(
        validation_alias=AliasChoices(
            "online_security",
            "OnlineSecurity"
        )
    )

    OnlineBackup: str = Field(
        validation_alias=AliasChoices(
            "online_backup",
            "OnlineBackup"
        )
    )

    DeviceProtection: str = Field(
        validation_alias=AliasChoices(
            "device_protection",
            "DeviceProtection"
        )
    )

    TechSupport: str = Field(
        validation_alias=AliasChoices(
            "tech_support",
            "TechSupport"
        )
    )

    # Streaming services

    StreamingTV: str = Field(
        validation_alias=AliasChoices(
            "streaming_tv",
            "StreamingTV"
        )
    )

    StreamingMovies: str = Field(
        validation_alias=AliasChoices(
            "streaming_movies",
            "StreamingMovies"
        )
    )

    # Contract and billing

    Contract: str = Field(
        validation_alias=AliasChoices(
            "contract",
            "Contract"
        )
    )

    PaperlessBilling: str = Field(
        validation_alias=AliasChoices(
            "paperless_billing",
            "PaperlessBilling"
        )
    )

    PaymentMethod: str = Field(
        validation_alias=AliasChoices(
            "payment_method",
            "PaymentMethod"
        )
    )


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(customer: Customer):

    customer_data = customer.model_dump()

    prediction, probability = predict_churn(
        customer_data
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