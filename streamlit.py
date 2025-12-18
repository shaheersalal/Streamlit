import streamlit as st
import pandas as pd
import joblib

model = joblib.load("bank_deposit_model.pkl")

def user_input_features():
    st.sidebar.header("Input Customer Data")

    balance = st.sidebar.number_input("balance", min_value=0, max_value=5000000, value=1000)
    duration = st.sidebar.number_input("duration", min_value=0, max_value=5000, value=100)
    housing = st.sidebar.selectbox("housing", ['yes', 'no'])
    month = st.sidebar.selectbox("month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    contact = st.sidebar.selectbox("contact", ['cellular', 'telephone', 'unknown'])
    default = st.sidebar.selectbox("default", [0, 1])
    marital = st.sidebar.selectbox("marital", ['married', 'single', 'divorced'])
    age = st.sidebar.number_input("age", min_value=10, max_value=100, value=30)
    
    data = {
        'balance': balance,
        'duration': duration,
        'housing': housing,
        'month': month,
        'contact': contact,
        'default': default,
        'marital': marital,        
        'age': age}
    

    features = pd.DataFrame([data])
    return features

def main():
    st.title("Bank Customer Prediction")

    input_df = user_input_features()

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Prediction: Customer will subscribe to term deposit. Probability: {prediction_proba:.2f}")
        else:
            st.error(f"Prediction: Customer will NOT subscribe to term deposit. Probability: {prediction_proba:.2f}")

if __name__ == "__main__":
    main()
