from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- FIX: ROBUST MODEL LOADING ---
# 1. Get the absolute path to the directory where api.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define potential model filenames based on your previous steps
# We check for both names to ensure it loads regardless of which task you ran last
possible_models = [
    os.path.join(BASE_DIR, '../models/best_fraud_model_v1.pkl'),
    os.path.join(BASE_DIR, '../models/random_forest_fraud_model.pkl')
]

model = None
for path in possible_models:
    if os.path.exists(path):
        print(f"Loading model from: {path}")
        model = joblib.load(path)
        break

if model is None:
    raise FileNotFoundError(f"Could not find a model file in ../models/. Checked: {possible_models}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Ensure the input is converted to a DataFrame with the correct features
        input_df = pd.DataFrame([data])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "status": "High Risk" if prediction == 1 else "Low Risk"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is running", "model_loaded": model is not None}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)