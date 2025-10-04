import pandas as pd
import numpy as np
import os

def create_engineered_features(filepath='data/cleaned_data.csv'):
    """
    Creates time-series, cyclical, and contextual features.
    """
    try:
        df = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        return {"error": "Cleaned data file not found. Please complete the cleaning step first."}

    energy_col = 'Energy Consumption (kWh)'
    if energy_col not in df.columns:
        return {"error": f"'{energy_col}' not found in the dataset."}

    # Select only numeric columns and resample to a consistent hourly frequency
    df_numeric = df.select_dtypes(include=np.number)
    df = df_numeric.resample('h').mean().interpolate(method='time')

    # --- Feature Creation ---
    features = {}

    # 1. Lag Features
    df['lag_1h'] = df[energy_col].shift(1)
    df['lag_24h'] = df[energy_col].shift(24)
    df['lag_7d'] = df[energy_col].shift(24 * 7)
    features['Lag Features'] = "Past energy consumption values (1 hour ago, 24 hours ago, 7 days ago) to capture short-term and weekly patterns."

    # 2. Moving Averages
    df['ma_24h'] = df[energy_col].rolling(window=24).mean()
    df['ma_7d'] = df[energy_col].rolling(window=24 * 7).mean()
    features['Moving Averages'] = "Average consumption over a rolling window (24 hours, 7 days) to smooth out noise and identify trends."

    # 3. Time Encodings
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    features['Time Encodings'] = "Cyclical and ordinal time features (hour, day of week, month) to help models understand time-based patterns."

    # 4. Fourier Seasonal Terms (captures seasonality better than simple month/day numbers)
    day_of_year = df.index.dayofyear
    year_length = 365.25
    df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / year_length)
    df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / year_length)
    features['Fourier Terms'] = "Sine and cosine transformations of time features (e.g., day of year) to represent seasonality in a continuous way that models can easily learn."
    
    # 5. Interaction Features
    temp_col = 'Temperature (°C)'
    if temp_col in df.columns:
        df['temp_x_hour_sin'] = df[temp_col] * np.sin(2 * np.pi * df['hour'] / 24)
        features['Interaction Features'] = "Combining two features to capture synergistic effects. Example: Temperature's impact might vary by the time of day."
    
    # 6. Context Features (Baseload vs. Peak)
    # Define peak hours (e.g., 8 AM to 8 PM)
    df['is_peak_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 20)).astype(int)
    features['Context Features'] = "Binary flags indicating context. Example: 'is_peak_hours' captures the difference in consumption between high-activity and low-activity periods."

    # --- Finalizing ---
    df.dropna(inplace=True)
    
    # Save the engineered dataset
    engineered_filepath = os.path.join('data', 'engineered_data.csv')
    df.to_csv(engineered_filepath)

    # Prepare a sample for the frontend
    # Rounding for cleaner display
    df_sample = df.head(100).round(2)
    df_sample.index = df_sample.index.strftime('%Y-%m-%d %H:%M')
    
    # Get all column names, including the new ones
    columns = df_sample.columns.tolist()
    data = [list(row) for row in df_sample.itertuples(index=True)] # Include index

    return {
        "features_explained": features,
        "sample_data": {
            "columns": ["Timestamp"] + columns,
            "data": data
        }
    }