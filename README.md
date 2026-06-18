# Customer Churn Prediction

## Overview

This machine learning project predicts whether a telecom customer is likely to churn based on customer information such as tenure, monthly charges, contract type, internet service, and payment method.

## Features

- Interactive Streamlit web application
- Real-time churn prediction
- Probability score for churn prediction
- User-friendly interface

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Streamlit
- Joblib

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
└── Customer Churn Prediction.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move into the project folder:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Input Features

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Gender
- Senior Citizen
- Partner Status

## Output

- Customer likely to churn
- Customer likely to stay
- Churn probability score

## Author

Shalini