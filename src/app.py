import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Import your modular classes
from preprocessing import FraudPreprocessor
from explainability import FraudExplainer

# Page configuration
st.set_page_config(page_title="Adey Innovations Fraud Monitor", layout="wide")

# Resolve absolute path to the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def load_assets():
    """Loads the model and a test sample for SHAP background data."""
    model_path = os.path.join(BASE_DIR, 'models', 'best_fraud_model.pkl')
    data_path = os.path.join(BASE_DIR, 'data', 'processed', 'X_test.csv')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
        
    model = joblib.load(model_path)
    # Background data for SHAP needs to be representative of the training set [cite: 168]
    test_sample = pd.read_csv(data_path).head(100) 
    return model, test_sample

# Load model and explainer
model, train_sample = load_assets()
explainer = FraudExplainer(model, train_sample)

# --- UI Layout ---
st.title("🛡️ Adey Innovations: Real-Time Fraud Detection")
st.markdown("Automated risk engine with transparent 'Reason Codes' for financial auditing[cite: 11, 63].")

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Transaction Details")
    # User inputs based on Fraud_Data.csv schema [cite: 27, 31, 36]
    purchase_val = st.number_input("Purchase Value ($)", value=120.0)
    age = st.number_input("User Age", min_value=18, max_value=100, value=30)
    source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
    browser = st.selectbox("Browser", ["Chrome", "Safari", "FireFox", "IE", "Opera", "Other"])
    sex = st.selectbox("Gender", ["M", "F"])

    if st.button("Analyze Risk"):
        # 1. Create a DataFrame for the raw input
        raw_input = pd.DataFrame([[purchase_val, age, source, browser, sex]], 
                                 columns=['purchase_value', 'age', 'source', 'browser', 'sex'])
        
        # 2. Add placeholders for missing engineered features (e.g. time-based) [cite: 134, 137]
        # In a full pipeline, these would be calculated from signup_time and purchase_time
        raw_input['time_since_signup'] = 0 
        raw_input['purchase_day'] = 0
        raw_input['purchase_hour'] = 0

        # 3. Handle Feature Alignment (One-Hot Encoding) [cite: 140, 166]
        # Convert categoricals to dummy variables
        input_encoded = pd.get_dummies(raw_input)
        
        # Get the exact feature names the model expects
        expected_features = model.feature_names_in_
        
        # Reindex the input to match model features, filling missing dummies with 0
        input_final = input_encoded.reindex(columns=expected_features, fill_value=0)

        # 4. Perform Prediction
        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0][1]

        with col2:
            st.subheader("Fraud Risk Assessment")
            if prediction == 1:
                st.error(f"🚨 FRAUD DETECTED (Risk Score: {probability:.2%})")
            else:
                st.success(f"✅ TRANSACTION SAFE (Risk Score: {probability:.2%})")
            
            # Show SHAP Explanation [cite: 60, 174]
            st.write("**Local Explanation (SHAP Force Plot):**")
            fig_force = explainer.explain_prediction(input_final)
            st.pyplot(fig_force)

# --- Global Insights ---
st.divider()
st.header("📈 Key Fraud Drivers")
st.write("Global feature importance across the dataset using SHAP[cite: 173].")
fig_summary = explainer.get_feature_importance_plot()
st.pyplot(fig_summary)