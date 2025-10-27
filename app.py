import sentry_sdk
sentry_sdk.init(
    dsn="https://59779dcff8aafa598ede055ada86bcb1@o4510262460481536.ingest.us.sentry.io/4510262500589568",
    traces_sample_rate=1.0
)
from prometheus_client import start_http_server, Summary, Counter

# Start Prometheus metrics on port 8000
start_http_server(8000)
PREDICTION_LATENCY = Summary('prediction_latency_seconds', 'Time spent predicting')
PREDICTION_COUNT = Counter('prediction_total', 'Total number of predictions made')

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.title("Customer Churn Prediction")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("churn_ann_model.h5")
    le_gender = pickle.load(open("le_gender.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, le_gender, scaler

model, le_gender, scaler = load_artifacts()

with st.form("user_input_form"):
    credit_score = st.number_input("Credit Score", value=650)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", value=40)
    tenure = st.number_input("Tenure", value=5)
    balance = st.number_input("Balance", value=50000)
    num_products = st.number_input("Num Of Products", value=2)
    has_card = st.selectbox("Has Credit Card (1=Yes, 0=No)", [1, 0])
    is_active = st.selectbox("Is Active Member (1=Yes, 0=No)", [1, 0])
    salary = st.number_input("Estimated Salary", value=90000)
    geo_germany = st.selectbox("Geography: Germany (1=Yes, 0=No)", [0, 1])
    geo_spain = st.selectbox("Geography: Spain (1=Yes, 0=No)", [0, 1])
    submitted = st.form_submit_button("Predict")

if submitted:
    gender_encoded = le_gender.transform([gender])[0]
    num_features = np.array([[credit_score, age, tenure, balance, num_products, salary]])
    num_scaled = scaler.transform(num_features).flatten()
    input_data = np.array([
        num_scaled[0],       # CreditScore (scaled)
        gender_encoded,      # Gender (encoded)
        num_scaled[1],       # Age (scaled)
        num_scaled[2],       # Tenure (scaled)
        num_scaled[3],       # Balance (scaled)
        num_scaled[4],       # NumOfProducts (scaled)
        has_card,            # HasCrCard (0/1)
        is_active,           # IsActiveMember (0/1)
        num_scaled[5],       # EstimatedSalary (scaled)
        geo_germany,         # Geography_Germany (0/1)
        geo_spain            # Geography_Spain (0/1)
    ]).reshape(1, -1)

    @PREDICTION_LATENCY.time()
    def predict_with_metrics(input_data):
        prob = float(model.predict(input_data)[0][0])
        PREDICTION_COUNT.inc()
        return prob

    prob = predict_with_metrics(input_data)
    pred = int(prob > 0.5)
    st.success(f"Churn prediction: **{'YES' if pred else 'NO'}** (Probability: {prob:.2f})")
