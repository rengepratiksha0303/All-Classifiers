import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(
    page_title="Breast Cancer Classifier",
    layout="wide"
)

st.image("BE-logo.png")

# -------------------------
# Load Dataset and Model
# -------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return data

@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

data = load_data()
model = load_model()

X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# -------------------------
# App Title
# -------------------------
st.title("🔬 Breast Cancer Prediction App")
st.write("Enter tumor measurements below to predict if the tumor is benign or malignant.")

# -------------------------
# Input Section
# -------------------------
st.header("Tumor Measurements")

input_data = {}
for feature in feature_names:
    idx = list(feature_names).index(feature)
    min_val = float(np.min(X[:, idx]))
    max_val = float(np.max(X[:, idx]))
    mean_val = float(np.mean(X[:, idx]))
    input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

input_df = pd.DataFrame([input_data])

st.subheader("Your Inputs")
st.dataframe(input_df)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    result = target_names[prediction]
    st.write(f"The model predicts that the tumor is **{result}**.")

    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame([prediction_proba], columns=target_names)
    st.dataframe(prob_df)

# -------------------------
# Dataset Info
# -------------------------
st.header("Dataset Info")
if st.checkbox("Show dataset"):
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    st.dataframe(df)
