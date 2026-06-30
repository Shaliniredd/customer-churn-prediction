import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and columns
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn")
st.caption("Model: Logistic Regression (class_weight='balanced', tuned via GridSearchCV)")

# --- User Inputs ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# --- Build input dataframe matching training columns ---
input_dict = {col: 0 for col in model_columns}

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

# Helper: safely set a one-hot column to 1 if it exists in model_columns
def set_flag(col_name):
    if col_name in input_dict:
        input_dict[col_name] = 1

if gender == "Male":
    set_flag('gender_Male')
if senior_citizen == "Yes":
    set_flag('SeniorCitizen_1')
if partner == "Yes":
    set_flag('Partner_Yes')
if dependents == "Yes":
    set_flag('Dependents_Yes')
if paperless_billing == "Yes":
    set_flag('PaperlessBilling_Yes')
if phone_service == "Yes":
    set_flag('PhoneService_Yes')

set_flag(f'MultipleLines_{multiple_lines}')
set_flag(f'OnlineSecurity_{online_security}')
set_flag(f'OnlineBackup_{online_backup}')
set_flag(f'DeviceProtection_{device_protection}')
set_flag(f'TechSupport_{tech_support}')
set_flag(f'StreamingTV_{streaming_tv}')
set_flag(f'StreamingMovies_{streaming_movies}')
set_flag(f'Contract_{contract}')
set_flag(f'InternetService_{internet_service}')
set_flag(f'PaymentMethod_{payment_method}')

input_df = pd.DataFrame([input_dict])
# keep column order identical to training
input_df = input_df[model_columns]

# Scale numeric columns (same scaler fitted during training)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# --- Predict ---
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay (Probability of churn: {probability:.2%})")