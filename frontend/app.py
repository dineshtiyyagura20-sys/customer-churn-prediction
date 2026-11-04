import streamlit as st
import requests


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "whether the customer is likely to churn."
)


# --------------------------------------------------
# API Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.5
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=846.0
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    customer_data = {

        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }

    try:

        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["churn_probability"]
            message = result["result"]

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "Stay"
                )

            if prediction == 1:

                st.error(
                    f"⚠️ {message}"
                )

            else:

                st.success(
                    f"✅ {message}"
                )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI server."
        )

        st.info(
            "Make sure FastAPI is running at "
            "http://127.0.0.1:8000"
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )