import pandas as pd
import os

def load_and_preview_data(filepath, num_rows=10):
    """
    Loads a CSV file into a pandas DataFrame and returns a preview.

    Args:
        filepath (str): The path to the CSV file.
        num_rows (int): The number of rows to include in the preview.

    Returns:
        dict: A dictionary containing the headers and the first N rows of data.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Define the correct path for the raw data snapshot
        save_path = os.path.join('data', 'raw_data.csv')
        
        # Save a raw data snapshot
        df.to_csv(save_path, index=False)
        
        preview = df.head(num_rows)
        return {
            'headers': preview.columns.tolist(),
            'rows': preview.to_dict(orient='records')
        }
    except FileNotFoundError:
        raise Exception("File not found at the specified path.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {e}")