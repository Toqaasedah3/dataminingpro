import os
import pandas as pd
from datasets import load_dataset

REPO_ID = "toqa66/hr-talent-mining-dataset"

RAW_FILE = "versions/hr_dataset_v0_raw.csv"
CLEANED_FILE = "prepared_data/cleaned_data.csv"
CLUSTERED_FILE = "prepared_data/clustered_data.csv"

PROCESSED_DIR = "backend/data/processed"


def _load_csv_from_huggingface(file_path: str) -> pd.DataFrame:
    dataset = load_dataset(
        REPO_ID,
        data_files=file_path
    )

    return dataset["train"].to_pandas()


def _save_to_processed(df: pd.DataFrame, filename: str) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    local_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(local_path, index=False)

    return local_path


def load_raw_dataset():
    df = _load_csv_from_huggingface(RAW_FILE)
    path = _save_to_processed(df, "hr_dataset_v0_raw.csv")

    return df, path


def load_cleaned_dataset():
    df = _load_csv_from_huggingface(CLEANED_FILE)
    path = _save_to_processed(df, "cleaned_data.csv")

    return df, path


def load_clustered_dataset():
    df = _load_csv_from_huggingface(CLUSTERED_FILE)
    path = _save_to_processed(df, "clustered_data.csv")

    return df, path