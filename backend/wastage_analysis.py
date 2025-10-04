import pandas as pd
import numpy as np
import os

def analyze_wastage_and_usage(data_path='data/engineered_data.csv'):
    """
    Analyzes energy wastage, classifies usage levels, and provides actionable insights.
    """
    try:
        df = pd.read_csv(data_path, index_col='Timestamp', parse_dates=True).sort_index()
    except FileNotFoundError:
        return {"error": f"Engineered data file not found at {data_path}. Please complete prior steps."}

    demand_col = 'Energy Consumption (kWh)'
    production_col = 'Local Energy Production (kWh)'
    
    if production_col not in df.columns:
        df[production_col] = 0
    if demand_col not in df.columns:
        return {"error": f"Required column '{demand_col}' not found for wastage analysis."}

    df['wastage'] = (df[production_col] - df[demand_col]).clip(lower=0)
    
    wastage_summary = df[['wastage', demand_col, production_col]].resample('D').sum()
    wastage_summary.to_excel('data/wastage_summary.xlsx')

    df['hour'] = df.index.hour
    hourly_avg_usage = df.groupby('hour')[demand_col].mean()
    
    high_use_threshold = hourly_avg_usage.quantile(0.80)
    low_use_threshold = hourly_avg_usage.quantile(0.20)

    high_use_hours = hourly_avg_usage[hourly_avg_usage >= high_use_threshold].index.tolist()
    low_use_hours = hourly_avg_usage[hourly_avg_usage <= low_use_threshold].index.tolist()

    top_10_wasted_days = wastage_summary.sort_values(by='wastage', ascending=False).head(10)
    
    # --- THIS IS THE FIX ---
    # Convert the Timestamp index of the KPI data into strings
    top_wasted_kpi = top_10_wasted_days['wastage']
    top_wasted_kpi.index = top_wasted_kpi.index.strftime('%Y-%m-%d')
    # --- END OF FIX ---

    calendar_data = wastage_summary[['wastage']].reset_index()
    calendar_data.rename(columns={'Timestamp': 'date'}, inplace=True)
    
    max_wastage = calendar_data['wastage'].max()
    calendar_data['intensity'] = calendar_data['wastage'] / max_wastage if max_wastage > 0 else 0
    
    insights = [
        "High energy wastage detected on weekends. Consider installing battery storage.",
        f"Peak energy usage occurs during hours: {high_use_hours}. Shift heavy loads to off-peak hours ({low_use_hours}).",
        "Wastage correlates with high solar irradiance. Align production with demand or storage.",
        "Review HVAC schedules during low-use periods to minimize consumption."
    ]

    return {
        "kpis": {
            "top_wasted_days": top_wasted_kpi.to_dict(), # Use the corrected data
            "high_use_hours": high_use_hours,
            "low_use_hours": low_use_hours,
        },
        "heatmap_data": calendar_data.to_dict(orient='records'),
        "insights": insights
    }