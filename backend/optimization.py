import pandas as pd
import numpy as np
import os
import time
from joblib import dump, load
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_optimization(filepath='data/selected_features.csv'):
    """
    Simulates hyperparameter optimization by training a model with pre-defined
    'optimized' parameters and compares it to the baseline model.
    """
    try:
        df = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True).sort_index()
    except FileNotFoundError:
        return {"error": "Selected features file not found. Please run prior steps first."}

    TARGET = 'Energy Consumption (kWh)'
    if TARGET not in df.columns:
        return {"error": f"Target column '{TARGET}' not found."}

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # --- 1. Load Baseline Model and Get "Before" Metrics ---
    try:
        # NOTE: This now loads the XGBoost model from your baseline_modeling.py script
        baseline_model_path = 'models/xgboost_energy_consumption_kwh_model.joblib'
        baseline_model = load(baseline_model_path)
    except Exception as e:
        return {"error": f"Could not load baseline XGBoost model: {e}. Please run Day 6 first."}

    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]

    test_size = int(len(df) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    baseline_preds = baseline_model.predict(X_test)
    before_metrics = {
        "MAE": round(mean_absolute_error(y_test, baseline_preds), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, baseline_preds)), 3),
        "R²": round(r2_score(y_test, baseline_preds), 3)
    }

    # --- 2. Define "Optimized" Parameters and Train New Model ---
    optimized_params = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }
    
    optimized_model = xgb.XGBRegressor(**optimized_params)
    optimized_model.fit(X_train, y_train)

    dump(optimized_model, 'models/xgboost_optimized_model.joblib')
    
    # --- 3. Get "After" Metrics ---
    optimized_preds = optimized_model.predict(X_test)
    after_metrics = {
        "MAE": round(mean_absolute_error(y_test, optimized_preds), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, optimized_preds)), 3),
        "R²": round(r2_score(y_test, optimized_preds), 3)
    }

    # --- 4. Calculate Improvement and Prepare Response ---
    improvement = {
        "MAE": round(((before_metrics['MAE'] - after_metrics['MAE']) / before_metrics['MAE']) * 100, 2),
        "RMSE": round(((before_metrics['RMSE'] - after_metrics['RMSE']) / before_metrics['RMSE']) * 100, 2),
        "R²": round(((after_metrics['R²'] - before_metrics['R²']) / (before_metrics['R²'] if before_metrics['R²'] != 0 else 1)) * 100, 2)
    }
    
    return {
        "model": "XGBoost",
        "before": before_metrics,
        "after": after_metrics,
        "improvement_percent": improvement
    }