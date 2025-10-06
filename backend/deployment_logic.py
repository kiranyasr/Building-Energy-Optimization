import pandas as pd
from joblib import load
import os

# --- Configuration & Model Loading ---
# This block runs once when the server starts. It tries to load the best model.
production_model = None
feature_list = []
MODEL_PATH = os.path.join('models', 'xgboost_optimized_model.joblib')
DATA_PATH = os.path.join('data', 'selected_features.csv')

try:
    # Attempt to load the final optimized model created in Day 8
    if os.path.exists(MODEL_PATH):
        production_model = load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Optimized model not found at {MODEL_PATH}. Please run the 'Optimization' step.")

    # Attempt to load the list of features the model was trained on
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
    # If any file is missing, we print a clear warning.
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
    
    # Convert the input dictionary into a pandas DataFrame with the correct column order
    input_df = pd.DataFrame([input_data], columns=feature_list)
    
    # Use the loaded model to make a prediction
    prediction = production_model.predict(input_df)
    
    # Return the prediction, rounded to 2 decimal places
    return round(prediction[0], 2)