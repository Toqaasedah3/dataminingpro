import os
import pandas as pd
from datasets import load_dataset
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.modules.preprocessing import clean_dataset, save_version, RAW_DIR, CLEANED_DIR
# from backend.modules.segmentation import run_segmentation
from backend.modules.column_detection import missing_important_columns
from backend.modules.huggingface_utils import upload_file_to_huggingface


app = FastAPI(title="HR Talent Mining API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CURRENT_DATA = {
    "raw_df": None,
    "cleaned_df": None,
    "encoded_df": None,
    "scaled_df": None,
    "detected_columns": None,
    "raw_path": None,
    "cleaned_path": None,
    "clustered_path": None,
}


HF_REPO_ID = "toqa66/hr-talent-mining-dataset"
HF_RAW_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


@app.get("/")
def root():
    return {"message": "HR Talent Mining API is running."}


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV file: {str(e)}"
        )

    clean_result = clean_dataset(df)

    cleaned_df = clean_result["cleaned_df"]
    encoded_df = clean_result["encoded_df"]
    scaled_df = clean_result["scaled_df"]
    detected_columns = clean_result["detected_columns"]

    raw_path = clean_result["saved_files"]["raw"]
    cleaned_path = clean_result["saved_files"]["cleaned"]

    CURRENT_DATA.update({
        "raw_df": df,
        "cleaned_df": cleaned_df,
        "encoded_df": encoded_df,
        "scaled_df": scaled_df,
        "detected_columns": detected_columns,
        "raw_path": raw_path,
        "cleaned_path": cleaned_path,
    })

    hf_raw = upload_file_to_huggingface(
        raw_path,
        f"versions/{os.path.basename(raw_path)}"
    )

    hf_cleaned = upload_file_to_huggingface(
        cleaned_path,
        f"versions/{os.path.basename(cleaned_path)}"
    )

    return {
        "status": "success",
        "source": "uploaded_csv",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_before_cleaning": int(df.isnull().sum().sum()),
        "missing_values_after_cleaning": clean_result["missing_values_after_cleaning"],
        "column_names": list(df.columns),
        "detected_columns": detected_columns,
        "missing_important_columns": missing_important_columns(detected_columns),
        "raw_version_path": raw_path,
        "cleaned_version_path": cleaned_path,
        "huggingface_raw": hf_raw,
        "huggingface_cleaned": hf_cleaned,
    }


@app.post("/load-from-huggingface")
def load_from_huggingface():
    try:
        dataset = load_dataset(
            HF_REPO_ID,
            data_files=HF_RAW_FILE
        )

        df = dataset["train"].to_pandas()

        clean_result = clean_dataset(df)

        cleaned_df = clean_result["cleaned_df"]
        encoded_df = clean_result["encoded_df"]
        scaled_df = clean_result["scaled_df"]
        detected_columns = clean_result["detected_columns"]

        raw_path = clean_result["saved_files"]["raw"]
        cleaned_path = clean_result["saved_files"]["cleaned"]

        CURRENT_DATA.update({
            "raw_df": df,
            "cleaned_df": cleaned_df,
            "encoded_df": encoded_df,
            "scaled_df": scaled_df,
            "detected_columns": detected_columns,
            "raw_path": raw_path,
            "cleaned_path": cleaned_path,
        })

        hf_cleaned = upload_file_to_huggingface(
            cleaned_path,
            f"versions/{os.path.basename(cleaned_path)}"
        )

        return {
            "status": "success",
            "source": "huggingface",
            "message": "Dataset loaded from Hugging Face and cleaned successfully.",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values_before_cleaning": int(df.isnull().sum().sum()),
            "missing_values_after_cleaning": clean_result["missing_values_after_cleaning"],
            "column_names": list(df.columns),
            "detected_columns": detected_columns,
            "missing_important_columns": missing_important_columns(detected_columns),
            "raw_version_path": raw_path,
            "cleaned_version_path": cleaned_path,
            "huggingface_cleaned": hf_cleaned,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while loading/cleaning/uploading dataset: {str(e)}"
        )

@app.get("/dataset-info")
def dataset_info():
    if CURRENT_DATA["raw_df"] is None:
        return {
            "status": "empty",
            "message": "No dataset uploaded or loaded yet."
        }

    df = CURRENT_DATA["raw_df"]

    return {
        "status": "available",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "column_names": list(df.columns),
        "detected_columns": CURRENT_DATA["detected_columns"],
        "raw_path": CURRENT_DATA["raw_path"],
        "cleaned_path": CURRENT_DATA["cleaned_path"],
        "clustered_path": CURRENT_DATA["clustered_path"],
    }