import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
DATA_FILE_PATH = 'data/selected_features.csv' 
MODELS_DIR = 'models/'
os.makedirs(MODELS_DIR, exist_ok=True)

TARGETS = ['Energy Consumption (kWh)', 'Lighting Consumption (kWh)', 'HVAC Consumption (kWh)'] 

# --- OPTIMIZATION: Simplified models for speed ---
MODELS = {
    'Ridge': Ridge(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=20, max_depth=5, random_state=42, n_jobs=-1),
}

def train_and_evaluate_models():
    """
    Loads, CLEANS, and trains simplified models on a small sample for speed.
    """
    if not os.path.exists(DATA_FILE_PATH):
        return {"error": f"Data file not found at {DATA_FILE_PATH}."}
    
    df = pd.read_csv(DATA_FILE_PATH, index_col='Timestamp', parse_dates=True)
    df.sort_index(inplace=True)

    # --- ROBUSTNESS FIX: Clean data one last time before training ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    all_results = {}

    for target in TARGETS:
        if target not in df.columns:
            continue
        
        X = df.drop(columns=TARGETS, errors='ignore')
        y = df[target]

        # Full time-series split to get a proper test set
        split_index = int(len(df) * 0.8)
        X_train_full, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train_full, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # --- SPEED OPTIMIZATION: Create a small sample for fast training ---
        train_df_full = X_train_full.join(y_train_full)
        sample_size = min(len(train_df_full), 3000) # Use a very small sample
        train_sample_df = train_df_full.sample(n=sample_size, random_state=42)
        
        X_train_sample = train_sample_df.drop(columns=[target])
        y_train_sample = train_sample_df[target]

        # Scale features based on the sample
        scaler = StandardScaler()
        X_train_scaled_sample = scaler.fit_transform(X_train_sample)
        X_test_scaled = scaler.transform(X_test)
        
        target_results = {
            "metrics": [],
            "predictions": {
                "index": X_test.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "actual": y_test.tolist()
            }
        }

        for name, model in MODELS.items():
            X_train_to_use = X_train_scaled_sample if name == 'Ridge' else X_train_sample
            X_test_to_use = X_test_scaled if name == 'Ridge' else X_test

            start_time = time.time()
            model.fit(X_train_to_use, y_train_sample) # Fit on the small sample
            training_time = time.time() - start_time

            y_pred = model.predict(X_test_to_use)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            model_filename = os.path.join(MODELS_DIR, f"{name.lower()}_{target.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.joblib")
            joblib.dump(model, model_filename)
            model_size = os.path.getsize(model_filename) / 1024

            target_results["metrics"].append({
                "Model": name, "MAE": f"{mae:.3f}", "RMSE": f"{rmse:.3f}", "R²": f"{r2:.3f}",
                "Training Time": f"{training_time:.2f} s", "Model Size": f"{model_size:.2f} KB"
            })
            target_results["predictions"][name] = y_pred.tolist()

        all_results[target] = target_results

    return all_results