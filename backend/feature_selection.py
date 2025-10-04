import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def select_and_reduce_features(filepath='data/engineered_data.csv'):
    """
    Performs feature selection and dimensionality reduction on a small,
    fast sample to ensure near-instantaneous results for the web UI.
    """
    try:
        # We only need to read a fraction of the data for the UI calculation
        # Read the full data later only when saving the final file.
        df_full = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        return {"error": "Engineered data file not found. Please complete the feature engineering step first."}

    # --- MAJOR OPTIMIZATION: Use a very small sample for UI calculations ---
    # This is the key to making the page load in under a second.
    sample_size = min(len(df_full), 10000) # Use 10,000 rows or less
    df_sample = df_full.sample(n=sample_size, random_state=42)

    # Define target and features from the sample
    TARGET = 'Energy Consumption (kWh)'
    if TARGET not in df_sample.columns:
        return {"error": f"Target column '{TARGET}' not found."}
    
    y_sample = df_sample[TARGET]
    X_sample = df_sample.drop(columns=[TARGET])

    # --- 1. Tree-based Feature Importance (Random Forest) ---
    # OPTIMIZATION: Use a much simpler model for extreme speed
    rf = RandomForestRegressor(
        n_estimators=10,      # Use very few trees
        max_depth=10,         # Limit tree depth
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_sample, y_sample)
    importances = sorted(zip(X_sample.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)

    # --- 2. Lightweight Dimensionality Reduction (PCA) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    # --- 3. Prepare Final Feature Set ---
    num_top_features = 15
    recommended_features = [feature for feature, importance in importances[:num_top_features]]
    
    # Use the FULL original dataset for the final output file
    df_selected = df_full[recommended_features + [TARGET]]
    
    # Save the final model-ready dataset
    selected_filepath = os.path.join('data', 'selected_features.csv')
    df_selected.to_csv(selected_filepath)
    
    # --- 4. Prepare JSON Response for Frontend ---
    response = {
        "rf_importances": {
            "features": [f[0] for f in importances[:20]],
            "scores": [f[1] for f in importances[:20]]
        },
        "pca_2d_visualization": {
            "x": X_pca_2d[:, 0].tolist(),
            "y": X_pca_2d[:, 1].tolist()
        },
        "recommended_set": recommended_features,
        "justification": f"Selected the top {num_top_features} features based on a rapid Random Forest model run on a representative sample of the data. This provides a near-instant and robust set of predictors for model training."
    }

    return response