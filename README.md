# Breast Cancer Classifier

A simple machine learning-based web application that predicts whether a tumor is **benign** or **malignant** using the scikit-learn breast cancer dataset and CatBoost model. Users can input values for all 30 tumor features and get real-time predictions.

---

## 🔗 Live Demo

Access the app here:  
➡️ [Breast Cancer Classifier - Streamlit App](https://all-classifiers-pratik-pranav.streamlit.app/)

---

## 📊 Flow Diagram

```mermaid
flowchart TD
    subgraph ML_Model_Creation
        A1[Load Dataset]
        A2[Preprocess Features]
        A3[Train CatBoost Model]
        A4[Save Model to File]
        
        A1 --> A2 --> A3 --> A4
    end

    subgraph App_Creation
        B1[Create Streamlit Interface]
        B2[Load Model File]
        B3[Take User Input]
        B4[Make Prediction]
        B5[Show Results]
        
        B1 --> B2 --> B3 --> B4 --> B5
    end

    A4 --> B2
