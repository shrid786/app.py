import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('diabetes_rf_model.pkl')

st.title("DIABITIES RISK PREDICTION HEURISTIC")
st.caption("INDIA IS ALSO CALLED DAIBITIES CAPITAL OF THE WORLD")

GLUCOSE = st.number_input("Glucose", value=148)
BODY_MASS_INDEX = st.number_input("BMI", value=33.6)
INSULIN= st.number_input("Insulin", value=125)
AGE = st.number_input("Age", value=50)

if st.button("Predict Risk"):
    # Format input exactly as done in Colab
    patient_data = pd.DataFrame([[glucose, bmi, insulin, age]], columns=['Glucose', 'BMI', 'Insulin', 'Age'])
    
    prediction = model.predict(patient_data)[0]
    prob = model.predict_proba(patient_data)[0][1]
    
    if prediction == 1:
        st.error(f"High risk of Diabetes detected (Probability: {prob:.1%})")
    else:
        st.success(f"Low risk of Diabetes (Probability: {prob:.1%})")
