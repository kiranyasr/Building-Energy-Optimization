import pandas as pd
import numpy as np
import os

def analyze_fluctuations(filepath='data/cleaned_data.csv'):
    """
    Computes fluctuation features, aggregates, and a volatility index.
    """
    try:
        # Load data with Timestamp as index
        df = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        return {"error": "Cleaned data file not found. Please complete the cleaning step first."}

    # Ensure we're working with the primary energy column
    energy_col = 'Energy Consumption (kWh)'
    if energy_col not in df.columns:
        return {"error": f"'{energy_col}' not found in the dataset."}

    # --- FIX: Select only numeric columns before resampling ---
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_numeric = df[numeric_cols]
    
    # Resample the numeric DataFrame to hourly, then sort
    df_resampled = df_numeric.resample('h').mean().sort_index()
    
    # Interpolate again after resampling to handle any gaps
    df_resampled[energy_col] = df_resampled[energy_col].interpolate(method='time')


    # --- 1. Compute Fluctuation Features (using the resampled dataframe) ---
    df_resampled['delta'] = df_resampled[energy_col].diff()
    df_resampled['pct_change'] = df_resampled[energy_col].pct_change() * 100

    # Rolling standard deviation for different windows (in hours for hourly data)
    df_resampled['rolling_std_24h'] = df_resampled[energy_col].rolling(window=24).std()
    df_resampled['rolling_std_7d'] = df_resampled[energy_col].rolling(window=24*7).std()
    df_resampled['rolling_std_30d'] = df_resampled[energy_col].rolling(window=24*30).std()


    # --- 2. Compute Volatility Index ---
    rolling_std = df_resampled['rolling_std_7d']
    df_resampled['volatility_index'] = (rolling_std - rolling_std.min()) / (rolling_std.max() - rolling_std.min()) * 100


    # --- 3. Add Aggregate Views ---
    daily_avg = df_resampled[energy_col].resample('D').mean()
    weekly_avg = df_resampled[energy_col].resample('W').mean()

    # --- 4. Save Features ---
    features_filepath = os.path.join('data', 'fluctuation_features.csv')
    df_resampled.to_csv(features_filepath)

    # --- 5. Prepare data for Frontend (JSON serializable) ---
    df_subset = df_resampled.last('365d').copy()

    # Convert timestamps to strings for JSON
    df_subset.index = df_subset.index.strftime('%Y-%m-%d %H:%M')
    daily_avg.index = daily_avg.index.strftime('%Y-%m-%d')
    weekly_avg.index = weekly_avg.index.strftime('%Y-%m-%d')
    
    # Replace NaN/inf with None for JSON compatibility
    df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_subset = df_subset.where(pd.notnull(df_subset), None)

    # Prepare final JSON payload
    response_data = {
        "kpis": {
            "current_volatility": round(df_subset['volatility_index'].iloc[-1], 2) if df_subset['volatility_index'].notna().any() else "N/A",
            "avg_volatility": round(df_subset['volatility_index'].mean(), 2) if df_subset['volatility_index'].notna().any() else "N/A",
            "avg_daily_consumption": round(daily_avg.mean(), 2)
        },
        "charts": {
            "pct_change": {
                "labels": df_subset.index.tolist(),
                "data": df_subset['pct_change'].tolist()
            },
            "rolling_std": {
                "labels": df_subset.index.tolist(),
                "data_24h": df_subset['rolling_std_24h'].tolist(),
                "data_7d": df_subset['rolling_std_7d'].tolist()
            },
            "aggregates": {
                "daily": {
                    "labels": daily_avg.index.tolist(),
                    "data": daily_avg.tolist()
                },
                "weekly": {
                    "labels": weekly_avg.index.tolist(),
                    "data": weekly_avg.tolist()
                }
            }
        }
    }

    return response_data