# app/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import feature_engineer

MODEL_PATH = "models/model.pkl"

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

# --- Input form ---
with st.form("passenger_form"):
    pclass = st.selectbox("Passenger class (Pclass)", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"])
    age = st.slider("Age", min_value=0, max_value=90, value=25)
    sibsp = st.number_input("Siblings/Spouse aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2)
    embarked = st.selectbox("Embarked", ["S","C","Q"])
    name = st.text_input("Full name (for Title)", "Mr Test")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a DataFrame in the same shape the training code expects
    data = pd.DataFrame([{
        "PassengerId": 0,
        "Pclass": pclass,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": None,
        "Fare": fare,
        "Cabin": None,
        "Embarked": embarked
    }])
    # Feature engineering (same as training)
    data = feature_engineer(data)
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','FamilySize','IsAlone','Deck']

    # Load model & predict
    model = joblib.load(MODEL_PATH)
    pred = model.predict(data[features])[0]
    proba = model.predict_proba(data[features])[0][1]

    st.metric("Survival probability", f"{proba:.2f}")
    if pred == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("💀 Prediction: Did not survive")
