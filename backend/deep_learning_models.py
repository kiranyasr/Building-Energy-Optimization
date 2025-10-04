import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Corrected Helper function that takes 3 arguments
def create_sequences(X_data, y_data, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        v = X_data[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_advanced_models(filepath='data/selected_features.csv'):
    try:
        df = pd.read_csv(filepath, index_col='Timestamp', parse_dates=True).sort_index()
    except FileNotFoundError:
        return {"error": "Selected features file not found. Please run prior steps."}

    TARGET = 'Energy Consumption (kWh)'
    if TARGET not in df.columns:
        return {"error": f"Target column '{TARGET}' not found."}

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    sample_size = min(len(df), 8000)
    df_sample = df.tail(sample_size).copy()
    
    y = df_sample[[TARGET]]
    X = df_sample.drop(columns=[TARGET])
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y)

    train_size = int(len(df_sample) * 0.8)
    
    # Create all scaled splits first
    train_X_scaled, test_X_scaled = X_scaled[:train_size], X_scaled[train_size:]
    train_y_scaled, test_y_scaled = y_scaled[:train_size], y_scaled[train_size:]
    
    N_STEPS = 24

    X_train_seq, y_train_seq = create_sequences(train_X_scaled, train_y_scaled, N_STEPS)
    X_test_seq, y_test_seq = create_sequences(test_X_scaled, test_y_scaled, N_STEPS)

    # --- 1. Standalone LSTM Model ---
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2), 
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history_lstm = lstm_model.fit(
        X_train_seq, y_train_seq, epochs=15, batch_size=32,
        validation_split=0.1, callbacks=[early_stopping], verbose=0
    )
    
    # --- 2. Hybrid Model (XGBoost + LSTM on Residuals) ---
    try:
        baseline_model_path = 'models/xgboost_energy_consumption_kwh_model.joblib'
        baseline_model = load(baseline_model_path)
    except Exception as e:
        return {"error": f"Could not load baseline XGBoost model: {e}. Please run Day 6 training first."}

    # Get baseline predictions on the unscaled training data
    gbm_train_preds = baseline_model.predict(X.iloc[:train_size])
    
    # Calculate residuals: (Scaled Actuals) - (Scaled Predictions)
    # The variable y_train was renamed to train_y_scaled earlier, this is the fix.
    residuals_train = train_y_scaled.flatten() - scaler_y.transform(gbm_train_preds.reshape(-1, 1)).flatten()
    
    # Create sequences using the scaled features and the calculated residuals
    X_res_seq, y_res_seq = create_sequences(train_X_scaled, residuals_train, N_STEPS)
    
    residual_lstm = Sequential([
        LSTM(32, activation='relu', input_shape=(X_res_seq.shape[1], X_res_seq.shape[2])),
        Dense(1)
    ])
    residual_lstm.compile(optimizer='adam', loss='mse')
    residual_lstm.fit(X_res_seq, y_res_seq, epochs=15, batch_size=32, verbose=0)
    
    # --- 3. Evaluate on Test Set ---
    lstm_preds_scaled = lstm_model.predict(X_test_seq, verbose=0)
    lstm_preds_orig = scaler_y.inverse_transform(lstm_preds_scaled).flatten()

    gbm_test_preds = baseline_model.predict(X.iloc[train_size:])
    residual_preds_scaled = residual_lstm.predict(X_test_seq, verbose=0)
    
    gbm_test_preds_scaled = scaler_y.transform(gbm_test_preds.reshape(-1, 1))
    hybrid_preds_scaled = gbm_test_preds_scaled[N_STEPS:] + residual_preds_scaled
    hybrid_preds_orig = scaler_y.inverse_transform(hybrid_preds_scaled).flatten()

    y_test_orig = scaler_y.inverse_transform(y_test_seq).flatten()
    baseline_preds_orig = gbm_test_preds[N_STEPS:]
    
    plot_labels = df_sample.index[train_size + N_STEPS:]

    return {"target": TARGET, "charts": {
        "loss_curves": {"loss": history_lstm.history['loss'], "val_loss": history_lstm.history['val_loss']},
        "predictions": {
            "labels": plot_labels.strftime('%Y-%m-%d %H:%M').tolist(),
            "actuals": y_test_orig.tolist(), "baseline": baseline_preds_orig.tolist(),
            "lstm": lstm_preds_orig.tolist(), "hybrid": hybrid_preds_orig.tolist()
        }}}