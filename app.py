import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and columns
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn")

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

# --- Build input dataframe matching training columns ---
input_dict = {col: 0 for col in model_columns}

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

if gender == "Male" and 'gender_Male' in input_dict:
    input_dict['gender_Male'] = 1
if senior_citizen == "Yes" and 'SeniorCitizen_1' in input_dict:
    input_dict['SeniorCitizen_1'] = 1
if partner == "Yes" and 'Partner_Yes' in input_dict:
    input_dict['Partner_Yes'] = 1

contract_col = f'Contract_{contract}'
if contract_col in input_dict:
    input_dict[contract_col] = 1

internet_col = f'InternetService_{internet_service}'
if internet_col in input_dict:
    input_dict[internet_col] = 1

payment_col = f'PaymentMethod_{payment_method}'
if payment_col in input_dict:
    input_dict[payment_col] = 1

input_df = pd.DataFrame([input_dict])

# Scale numeric columns
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