import numpy as np
import pandas as pd
import shap
import joblib
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/best_transformer_mev_model.h5')

# Load the scaler
scaler = joblib.load('models/assets/scaler.pkl')

# Load the feature names
with open('models/assets/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# Load the test data
X_test = np.load('models/assets/X_test.npy')
y_test = np.load('models/assets/y_test.npy')

# Flatten it back for SHAP
X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[2]))

# Define a wrapper function to handle the reshaping for the model prediction
def model_predict_reshaped(data):
    reshaped_data = data.reshape((data.shape[0], 1, data.shape[1]))
    return model(reshaped_data)

# Create a SHAP explainer using Explainer with TensorFlow's model
explainer = shap.Explainer(model_predict_reshaped, X_test_flat)

# Calculate SHAP values
shap_values = explainer(X_test_flat)

# Save the SHAP values and expected values
joblib.dump(shap_values.values, 'models/assets/shap_values.npy')
joblib.dump(shap_values.base_values, 'models/assets/expected_values.npy')
joblib.dump(X_test_flat, 'models/assets/X_test_flat.npy')
joblib.dump(feature_names, 'models/assets/feature_names.pkl')
joblib.dump(X_test_flat, 'models/assets/X_test_flat.pkl')