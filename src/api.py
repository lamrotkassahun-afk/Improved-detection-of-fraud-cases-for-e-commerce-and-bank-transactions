from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# --- Path Configuration ---
# Resolves absolute path to the project root to find the models folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_fraud_model.pkl')

# Load the model once at startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"ERROR: Model file not found at {MODEL_PATH}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for real-time fraud inference.
    Expects JSON input: {"purchase_value": 120, "age": 30, "source": "Ads", ...}
    """
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert incoming JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # 1. Feature Alignment Logic
        # Get the exact feature names the model was trained on
        expected_features = model.feature_names_in_
        
        # Perform one-hot encoding on categoricals (browser, source, sex)
        input_encoded = pd.get_dummies(input_df)
        
        # Reindex to match the model's schema, filling missing dummy columns with 0
        # This creates 'input_final' correctly for the model
        input_final = input_encoded.reindex(columns=expected_features, fill_value=0)
        
        # 2. Inference
        prediction = int(model.predict(input_final)[0])
        probability = float(model.predict_proba(input_final)[0][1])
        
        # 3. Return JSON Response
        return jsonify({
            'fraud_prediction': prediction,
            'risk_score': round(probability, 4),
            'is_fraud': bool(prediction),
            'status': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Failed'}), 500

# Health check route
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': 'Adey Innovations Fraud API is active', 'status': 'Active'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)