import pandas as pd
import io

# Global in-memory data store (can be replaced with DB if needed)
df_storage = {}

def save_dataframe(key, df):
    df_storage[key] = df

def get_dataframe(key):
    return df_storage.get(key)

def read_csv_from_upload(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def serialize_dataframe(df):
    """Return a preview of a dataframe for frontend."""
    return {
        "columns": df.columns.tolist(),
        "preview": df.head().to_dict(orient="records")
    }

def to_csv_response(df):
    """Converts a DataFrame to CSV string for download."""
    return df.to_csv(index=False)
