import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from backend.modules.column_detection import detect_hr_columns


PROCESSED_DIR = os.path.join("backend", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_dataset(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty.")

    raw_df = df.copy()
    cleaned_df = df.copy()

    # 1. Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # 2. Fix column names
    cleaned_df.columns = [
        str(col).strip()
        for col in cleaned_df.columns
    ]

    # 3. Fill missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            mode_value = cleaned_df[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
        else:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(
                median_value if pd.notna(median_value) else 0
            )

    # 4. Detect HR columns
    detected_columns = detect_hr_columns(cleaned_df)

    # 5. Encode categorical columns
    encoded_df = cleaned_df.copy()

    for col in encoded_df.columns:
        if encoded_df[col].dtype == "object":
            unique_values = encoded_df[col].nunique()

            if unique_values == 2:
                values = list(encoded_df[col].dropna().unique())
                mapping = {values[0]: 0, values[1]: 1}
                encoded_df[col] = encoded_df[col].map(mapping)
            else:
                encoded_df = pd.get_dummies(
                    encoded_df,
                    columns=[col],
                    drop_first=True
                )

    encoded_df = encoded_df.apply(pd.to_numeric, errors="coerce")
    encoded_df = encoded_df.fillna(0)

    # 6. Scale numeric data
    scaled_df = encoded_df.copy()

    numeric_cols = scaled_df.select_dtypes(include=["int64", "float64", "bool"]).columns

    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        scaled_df[numeric_cols] = scaler.fit_transform(
            scaled_df[numeric_cols]
        )

    # 7. Save outputs
    raw_path = os.path.join(PROCESSED_DIR, "hr_dataset_v0_raw.csv")
    cleaned_path = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    encoded_path = os.path.join(PROCESSED_DIR, "encoded_data.csv")
    scaled_path = os.path.join(PROCESSED_DIR, "scaled_data.csv")

    raw_df.to_csv(raw_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)
    encoded_df.to_csv(encoded_path, index=False)
    scaled_df.to_csv(scaled_path, index=False)

    return {
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "encoded_df": encoded_df,
        "scaled_df": scaled_df,
        "detected_columns": detected_columns,
        "missing_values_after_cleaning": int(cleaned_df.isnull().sum().sum()),
        "saved_files": {
            "raw": raw_path,
            "cleaned": cleaned_path,
            "encoded": encoded_path,
            "scaled": scaled_path,
        }
    }