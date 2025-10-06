import pandas as pd
from joblib import load
import os
import numpy as np

# --- Configuration & Model Loading ---
production_model = None
feature_list = []
MODEL_PATH = os.path.join('models', 'xgboost_optimized_model.joblib')
DATA_PATH = os.path.join('data', 'selected_features.csv')

try:
    if os.path.exists(MODEL_PATH):
        production_model = load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Optimized model not found at {MODEL_PATH}. Please run the 'Optimization' step.")

    if os.path.exists(DATA_PATH):
        data_df = pd.read_csv(DATA_PATH, index_col='Timestamp')
        target_col = 'Energy Consumption (kWh)'
        if target_col in data_df.columns:
            feature_list = data_df.columns.drop(target_col).tolist()
        else:
            feature_list = data_df.columns.tolist()
    else:
        raise FileNotFoundError(f"Feature list file not found at {DATA_PATH}. Please run the 'Feature Selection' step.")

except Exception as e:
    print(f"WARNING: Could not initialize deployment logic. Error: {e}")


def get_feature_list():
    """Returns the list of features the model needs."""
    if not feature_list:
        raise RuntimeError("Feature list is not available. Ensure the 'Feature Selection' step has been run successfully.")
    return feature_list


def make_prediction(input_data):
    """Takes a dictionary of input data and returns a model prediction."""
    if production_model is None:
        raise RuntimeError("Model is not loaded. Please run the 'Optimization' step to create the final model.")
    
    # Convert input to DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=feature_list)
    
    # Make prediction
    prediction = production_model.predict(input_df)
    
    # Convert NumPy float32/float64 to native Python float
    py_float = float(np.round(prediction[0], 2))
    
    return py_float
