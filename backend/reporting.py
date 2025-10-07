# reporting.py
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compile_final_report():
    """
    Compiles a final report comparing all trained models on a test set.
    """
    os.makedirs("data", exist_ok=True)
    try:
        df = pd.read_csv('data/selected_features.csv', index_col='Timestamp', parse_dates=True).sort_index()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
    except FileNotFoundError:
        return {"error": "Selected features file not found. Please complete prior steps."}

    TARGET = 'Energy Consumption (kWh)'
    if TARGET not in df.columns:
        return {"error": f"Target column '{TARGET}' not found."}

    features = [col for col in df.columns if col != TARGET]
    X = df[features]
    y = df[TARGET]
    test_size = int(len(df) * 0.2)
    X_test, y_test = X.iloc[-test_size:], y.iloc[-test_size:]

    model_paths = {
        "Ridge": "models/ridge_energy_consumption_kwh_model.joblib",
        "RandomForest": "models/randomforest_energy_consumption_kwh_model.joblib",
        "XGBoost (Baseline)": "models/xgboost_energy_consumption_kwh_model.joblib",
        "XGBoost (Optimized)": "models/xgboost_optimized_model.joblib"
    }

    all_metrics = []

    for name, path in model_paths.items():
        try:
            model = load(path)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            model_size_kb = os.path.getsize(path) / 1024

            all_metrics.append({
                "Model": name,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R²": round(r2, 3),
                "Model Size (KB)": round(model_size_kb, 2)
            })
        except FileNotFoundError:
            print(f"Warning: Model file not found for {name}. Skipping.")
            continue

    if not all_metrics:
        return {"error": "No trained models found to compare. Run baseline and advanced models first."}

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_excel("data/comparison_summary.xlsx", index=False)

    chart_data = {
        "labels": summary_df["Model"].tolist(),
        "mae_data": summary_df["MAE"].tolist(),
        "rmse_data": summary_df["RMSE"].tolist(),
        "r2_data": summary_df["R²"].tolist()
    }

    # Feature importance for optimized XGBoost
    if "XGBoost (Optimized)" not in model_paths or not os.path.exists(model_paths["XGBoost (Optimized)"]):
        return {"error": "Optimized XGBoost model not found."}

    best_model = load(model_paths["XGBoost (Optimized)"])
    importances = best_model.feature_importances_
    try:
        feature_names = best_model.get_booster().feature_names
    except AttributeError:
        feature_names = X_test.columns.tolist()

    importance_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]

    response = {
        "summary_table": all_metrics,
        "charts": {
            "accuracy_comparison": chart_data,
            "feature_importance": {
                "features": [x[0] for x in importance_data],
                "scores": [float(x[1]) for x in importance_data]
            }
        }
    }

    return response
