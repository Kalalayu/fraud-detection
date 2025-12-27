import os
import pandas as pd

def load_data(filename):
    file_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    return pd.read_csv(file_path)
