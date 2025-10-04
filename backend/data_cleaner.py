import pandas as pd
import numpy as np
import os

def clean_and_summarize_data(filepath='data/uploaded_data.csv'):
    """
    Performs a full cleaning process on the dataset and returns a summary.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return {"error": "Uploaded data file not found. Please upload a file first."}

    # --- 1. Initial Inspection & Datetime Parsing ---
    initial_rows = len(df)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # Drop rows where timestamp could not be parsed
    df.dropna(subset=['Timestamp'], inplace=True)
    df.set_index('Timestamp', inplace=True)

    # --- 2. Handle Missing Values ---
    missing_before = df.isnull().sum().sum()
    # Using forward fill for time-series data
    df.interpolate(method='time', limit_direction='forward', inplace=True)
    df.fillna(method='bfill', inplace=True) # Backfill for any remaining NaNs at the start
    missing_after = df.isnull().sum().sum()
    missing_fixed = missing_before - missing_after


    # --- 3. Handle Duplicates ---
    duplicates_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    duplicates_removed = duplicates_before


    # --- 4. Cast Data Types (identify numeric columns) ---
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()


    # --- 5. Outlier Detection (IQR Method) ---
    outliers_found = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            outliers_found[col] = int(outlier_count)
            # Flagging outliers instead of removing
            df[f'{col}_is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    total_outliers = sum(outliers_found.values())

    # --- 6. Save Cleaned Data ---
    cleaned_filepath = os.path.join('data', 'cleaned_data.csv')
    df.to_csv(cleaned_filepath)
    
    # --- 7. Generate Summary ---
    summary = {
        "initial_rows": int(initial_rows),
        "rows_after_cleaning": len(df),
        "missing_values_fixed": int(missing_fixed),
        "duplicates_removed": int(duplicates_removed),
        "outliers_found": outliers_found,
        "total_outliers_flagged": int(total_outliers),
        "cleaned_filepath": cleaned_filepath
    }

    return summary