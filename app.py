
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and training feature names
model = joblib.load("churn_model.pkl")
features = joblib.load("churn_features.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# Customer details
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)
contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, value=70.0
)
total_charges = st.number_input(
    "Total Charges ($)", min_value=0.0, value=840.0
)

if st.button("Predict Churn"):

    # Create customer data
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
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
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Convert categories into model features
    customer = pd.get_dummies(customer, drop_first=True, dtype=int)

    # Match the training feature columns
    customer = customer.reindex(columns=features, fill_value=0)

    # Predict probability and apply 40% threshold
    probability = model.predict_proba(customer)[0][1]
    threshold = 0.40

    st.subheader("Prediction Result")
    st.write(f"Estimated churn probability: {probability:.1%}")

    if probability >= threshold:
        st.error("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted not to churn.")
