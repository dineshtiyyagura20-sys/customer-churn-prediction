# Customer Churn Prediction – MLOps

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to **churn (leave the service)** or **stay** based on customer information, service usage, contract details, and billing information.

The project follows an end-to-end **MLOps workflow**, including data preprocessing, model training, model evaluation, prediction, API development, containerization, and cloud deployment.

Multiple machine learning models are trained and compared to identify the best-performing model. The final application provides real-time predictions through a **FastAPI backend** and an interactive **Streamlit frontend**.

The application is containerized using **Docker and Docker Compose** and deployed on **AWS EC2**.

---

## 🎯 Project Objectives

- Predict customer churn using machine learning.
- Compare multiple classification algorithms.
- Select the best-performing model.
- Provide real-time churn predictions.
- Build a REST API using FastAPI.
- Create an interactive dashboard using Streamlit.
- Containerize the application using Docker.
- Deploy the application on AWS EC2.
- Follow an MLOps-oriented project structure.

---

## 🤖 Machine Learning Models

The following models are used:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

The models are trained and evaluated using multiple performance metrics.

---

## 🏆 Best Model

Based on **accuracy**, the best-performing model is:

**Logistic Regression**

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **80.70%** | 65.84% | 56.68% | 60.92% | 84.17% |
| Decision Tree | 80.06% | 66.67% | 49.73% | 56.97% | 82.55% |
| Random Forest | 80.20% | 66.32% | 51.60% | 58.05% | **84.35%** |

**Best model by accuracy: Logistic Regression – 80.70%**

> Note: Random Forest achieved the highest ROC-AUC score (84.35%), while Logistic Regression achieved the highest accuracy (80.70%).

---

## 🏗️ Project Architecture

```text
Customer Churn Prediction
│
├── Data
│   └── Customer Dataset
│
├── Data Preprocessing
│   ├── Data Cleaning
│   ├── Missing Value Handling
│   └── Feature Processing
│
├── Machine Learning
│   ├── Logistic Regression
│   ├── Decision Tree
│   └── Random Forest
│
├── Model Evaluation
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1 Score
│   └── ROC-AUC
│
├── FastAPI
│   └── Prediction API
│
├── Streamlit
│   └── User Interface
│
├── Docker
│   ├── API Container
│   └── Frontend Container
│
└── AWS EC2
    └── Production Deployment
```

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── api/
│   └── main.py
│
├── artifacts/
│   └── model/
│       ├── model_comparison.csv
│       └── trained model files
│
├── data/
│   └── customer churn dataset
│
├── frontend/
│   └── streamlit_app.py
│
├── notebooks/
│   └── Jupyter notebooks
│
├── src/
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── other source files
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_analysis.py
│   └── test_validation.py
│
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── params.yaml
├── dvc.yaml
├── pytest.ini
└── README.md
```

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Logistic Regression
- Decision Tree
- Random Forest

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### MLOps / DevOps
- Docker
- Docker Compose
- DVC
- Git
- GitHub

### Cloud
- AWS EC2

---

## 🔄 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Prediction
```

---

## 🚀 Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/dineshtiyyagura20-sys/customer-churn-prediction.git
```

### 2. Navigate to the Project

```bash
cd customer-churn-prediction
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

---

## 🎨 Run Streamlit Frontend

```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit application will normally be available at:

```text
http://localhost:8501
```

---

## 🐳 Run Using Docker Compose

The application contains two Docker services:

- FastAPI backend
- Streamlit frontend

Build and start the containers:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker ps
```

Stop the application:

```bash
docker compose down
```

---

## ☁️ AWS EC2 Deployment

The application is deployed on an **AWS EC2 instance** using Docker and Docker Compose.

Deployment flow:

```text
GitHub Repository
       ↓
AWS EC2
       ↓
Docker
       ↓
Docker Compose
       ↓
FastAPI + Streamlit
       ↓
Live Application
```

### Deployment Command

```bash
git clone https://github.com/dineshtiyyagura20-sys/customer-churn-prediction.git
cd customer-churn-prediction
docker compose up -d --build
```

---

## 🌐 Live Application

### Streamlit Application

**Live Demo:**

http://107.23.161.209:8501

### API

```text
http://107.23.161.209:8000
```

### FastAPI Swagger Documentation

```text
http://107.23.161.209:8000/docs
```

---

## 📊 Example Prediction

The application accepts customer information such as:

- Senior Citizen
- Tenure
- Monthly Charges
- Total Charges
- Gender
- Partner
- Dependents
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method

The model returns:

```text
Churn Probability: 13.10%

Prediction: Stay

Customer is likely to stay.
```

---

## 📈 Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified customers.

### Precision

Measures how many customers predicted as churn actually churn.

### Recall

Measures how many actual churn customers were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers.

---

## 💡 Business Problem

Customer churn is an important business problem because losing customers can reduce revenue and increase customer acquisition costs.

This project helps businesses identify customers who are more likely to leave so that they can take proactive actions such as:

- Customer retention campaigns
- Personalized offers
- Improved customer support
- Contract incentives
- Targeted communication

---

## 🔮 Future Improvements

- Implement automated model retraining.
- Add MLflow experiment tracking.
- Add CI/CD pipeline using GitHub Actions.
- Add model monitoring.
- Add data drift detection.
- Improve model performance through hyperparameter tuning.
- Add authentication to the API.
- Use HTTPS with a domain name.
- Deploy using AWS ECS or Kubernetes.
- Add automated testing to the deployment pipeline.

---

## 👨‍💻 Author

**DINESH REDDY TIYYAGURA**

B.Tech – Computer Science and Engineering

GitHub:  
https://github.com/dineshtiyyagura20-sys

---

## 📄 License

This project is created for educational and portfolio purposes.
