import os
from datetime import datetime
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.modules.column_detection import detect_hr_columns


RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"


def save_version(df: pd.DataFrame, folder: str, version_name: str) -> str:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder, f"{version_name}_{timestamp}.csv")
    df.to_csv(file_path, index=False)
    return file_path


def auto_clean_hr_data(df: pd.DataFrame):
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop_duplicates()

    for col in cleaned_df.columns:
        if is_numeric_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        else:
            mode_value = cleaned_df[col].mode()
            cleaned_df[col] = cleaned_df[col].fillna(
                mode_value[0] if not mode_value.empty else "Unknown"
            )

    for col in cleaned_df.columns:
        if not is_numeric_dtype(cleaned_df[col]):
            values = set(cleaned_df[col].astype(str).str.lower().str.strip().unique())

            if values.issubset({"yes", "no"}):
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip().map({
                    "yes": 1,
                    "no": 0
                })

            elif values.issubset({"true", "false"}):
                cleaned_df[col] = cleaned_df[col].astype(str).str.lower().str.strip().map({
                    "true": 1,
                    "false": 0
                })

    encoded_df = cleaned_df.copy()

    for col in encoded_df.select_dtypes(include=["object", "category", "string"]).columns:
        encoder = LabelEncoder()
        encoded_df[col] = encoder.fit_transform(encoded_df[col].astype(str))

    scaled_df = encoded_df.copy()
    numeric_cols = scaled_df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns

    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        scaled_df[numeric_cols] = scaler.fit_transform(scaled_df[numeric_cols])

    return {
        "cleaned_df": cleaned_df,
        "encoded_df": encoded_df,
        "scaled_df": scaled_df,
        "missing_values_after_cleaning": int(cleaned_df.isnull().sum().sum())
    }


def clean_dataset(df: pd.DataFrame):
    result = auto_clean_hr_data(df)

    cleaned_df = result["cleaned_df"]
    encoded_df = result["encoded_df"]
    scaled_df = result["scaled_df"]
    detected_columns = detect_hr_columns(df)

    raw_path = save_version(df, RAW_DIR, "raw_dataset")
    cleaned_path = save_version(cleaned_df, CLEANED_DIR, "cleaned_dataset")
    encoded_path = save_version(encoded_df, CLEANED_DIR, "encoded_dataset")
    scaled_path = save_version(scaled_df, CLEANED_DIR, "scaled_dataset")

    return {
        "cleaned_df": cleaned_df,
        "encoded_df": encoded_df,
        "scaled_df": scaled_df,
        "detected_columns": detected_columns,
        "missing_values_after_cleaning": result["missing_values_after_cleaning"],
        "saved_files": {
            "raw": raw_path,
            "cleaned": cleaned_path,
            "encoded": encoded_path,
            "scaled": scaled_path
        }
    }