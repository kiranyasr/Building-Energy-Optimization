import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.ensemble import IsolationForest

def detect_anomalies(model_path='models/xgboost_optimized_model.joblib', data_path='data/engineered_data.csv'):
    """
    Uses the optimized model to find anomalies in the full dataset.
    """
    # --- 1. Load Model and Full Feature Data ---
    try:
        model = load(model_path)
        df = pd.read_csv(data_path, index_col='Timestamp', parse_dates=True).sort_index()
    except FileNotFoundError as e:
        return {"error": f"File not found: {e}. Please ensure prior steps are complete."}

    TARGET = 'Energy Consumption (kWh)'
    if TARGET not in df.columns:
        return {"error": f"Target column '{TARGET}' not found."}

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # --- 2. Calculate Residuals ---
    features = model.get_booster().feature_names
    X = df[features]
    y_true = df[TARGET]
    
    y_pred = model.predict(X)
    
    df['residual'] = y_true - y_pred
    
    # --- 3. Detect Anomalies ---
    # Method 1: Statistical (Z-score)
    mean_res = df['residual'].mean()
    std_res = df['residual'].std()
    df['z_score'] = (df['residual'] - mean_res) / std_res
    df['is_anomaly_zscore'] = df['z_score'].abs() > 3 # Flag anything beyond 3 standard deviations

    # Method 2: Machine Learning (Isolation Forest)
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df['is_anomaly_isoforest'] = iso_forest.fit_predict(df[['residual']])
    df['is_anomaly_isoforest'] = df['is_anomaly_isoforest'] == -1 # Convert -1/1 to True/False

    # Combine results
    df['is_anomaly'] = df['is_anomaly_zscore'] | df['is_anomaly_isoforest']
    
    anomalies = df[df['is_anomaly']].copy()
    anomalies['anomaly_type'] = np.where(anomalies['residual'] > 0, 'Spike', 'Dip')
    
    # --- 4. Save Anomaly Log ---
    anomaly_log = anomalies[['residual', 'anomaly_type']].reset_index()
    anomaly_log.to_csv('data/anomaly_log.csv', index=False)
    
    # --- 5. Prepare Data for Frontend ---
    # a. For Residual Plot
    anomaly_points = anomalies.reset_index()[['Timestamp', 'residual']].to_dict(orient='records')

    # b. For Anomaly Table (show most recent 100)
    anomaly_table_data = anomaly_log.tail(100).to_dict(orient='records')

    # c. For Heatmap
    heatmap_data = df.reset_index()[['Timestamp', 'residual']]
    heatmap_data['date'] = heatmap_data['Timestamp'].dt.date
    daily_severity = heatmap_data.groupby('date')['residual'].apply(lambda x: x.abs().sum()).reset_index()
    daily_severity.rename(columns={'residual': 'severity'}, inplace=True)
    
    response = {
        "residual_plot": {
            "labels": df.index.strftime('%Y-%m-%d %H:%M').tolist(),
            "residuals": df['residual'].tolist(),
            "anomalies": anomaly_points
        },
        "anomaly_table": anomaly_table_data,
        "heatmap_data": daily_severity.to_dict(orient='records')
    }
    
    return response