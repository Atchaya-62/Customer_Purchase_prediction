import streamlit as st
import joblib
import pandas as pd


pipeline = joblib.load("knn_ads_pipeline.pkl")

st.set_page_config(page_title="KNN Social Ads Predictor", page_icon="📊", layout="centered")

st.title("📊 Social Network Ads Purchase Prediction")
st.write("Predict whether a user will purchase a product based on Age, Gender, and Estimated Salary.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=80, value=30)
salary = st.number_input("Estimated Salary", min_value=10000, max_value=200000, value=50000)



if st.button("Predict"):
    
    user_input = {
    "age": age,
    "estimated_salary": salary,
    "gender": gender
    }

    user_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(user_df)[0]


    if prediction == 1:
        st.success("🟢 The user is **likely to purchase** the product.")
    else:
        st.warning("🔴 The user is **unlikely to purchase** the product.")
