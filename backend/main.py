import os
import pandas as pd
from datasets import load_dataset
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.modules.recommendation_engine import generate_recommendations
from backend.modules.risk_detection import run_risk_detection
from backend.modules.preprocessing import clean_dataset
from backend.modules.segmentation import run_segmentation
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
    "encoded_path": None,
    "scaled_path": None,
    "clustered_path": None,
    "segmentation_summary": None,
    "segmentation_metrics": None,
    "risk_path": None,
}


HF_REPO_ID = "toqa66/hr-talent-mining-dataset"
HF_RAW_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"


@app.get("/")
def root():
    return {"message": "HR Talent Mining API is running."}


def process_dataset(df: pd.DataFrame):
    clean_result = clean_dataset(df)

    CURRENT_DATA.update({
        "raw_df": df,
        "cleaned_df": clean_result["cleaned_df"],
        "encoded_df": clean_result["encoded_df"],
        "scaled_df": clean_result["scaled_df"],
        "detected_columns": clean_result["detected_columns"],
        "raw_path": clean_result["saved_files"]["raw"],
        "cleaned_path": clean_result["saved_files"]["cleaned"],
        "encoded_path": clean_result["saved_files"]["encoded"],
        "scaled_path": clean_result["saved_files"]["scaled"],
        "clustered_path": None,
        "segmentation_summary": None,
        "segmentation_metrics": None,
        "risk_path": None,
    })

    return clean_result


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        df = pd.read_csv(file.file)
        clean_result = process_dataset(df)

        return {
            "status": "success",
            "source": "uploaded_csv",
            "message": "Dataset uploaded, cleaned, encoded, and scaled successfully.",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values_before_cleaning": int(df.isnull().sum().sum()),
            "missing_values_after_cleaning": clean_result["missing_values_after_cleaning"],
            "column_names": list(df.columns),
            "detected_columns": clean_result["detected_columns"],
            "missing_important_columns": missing_important_columns(
                clean_result["detected_columns"]
            ),
            "saved_files": clean_result["saved_files"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while uploading/cleaning dataset: {str(e)}"
        )


@app.post("/load-from-huggingface")
def load_from_huggingface():
    try:
        dataset = load_dataset(
            HF_REPO_ID,
            data_files=HF_RAW_FILE
        )

        df = dataset["train"].to_pandas()
        clean_result = process_dataset(df)

        return {
            "status": "success",
            "source": "huggingface",
            "message": "Dataset loaded from Hugging Face, cleaned, encoded, and scaled successfully.",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values_before_cleaning": int(df.isnull().sum().sum()),
            "missing_values_after_cleaning": clean_result["missing_values_after_cleaning"],
            "column_names": list(df.columns),
            "detected_columns": clean_result["detected_columns"],
            "missing_important_columns": missing_important_columns(
                clean_result["detected_columns"]
            ),
            "saved_files": clean_result["saved_files"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while loading/cleaning dataset: {str(e)}"
        )


@app.post("/run-segmentation")
def segmentation_api():
    if CURRENT_DATA["cleaned_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset available. Please upload or load a dataset first."
        )

    try:
        result = run_segmentation(
            scaled_df=CURRENT_DATA["scaled_df"],
            original_df=CURRENT_DATA["cleaned_df"]
        )

        CURRENT_DATA["clustered_path"] = result["output_path"]
        CURRENT_DATA["segmentation_summary"] = result["summary"]

        CURRENT_DATA["segmentation_metrics"] = {
            "best_k": result["best_k"],
            "features_used": result["features_used"],
            "silhouette_scores": result["silhouette_scores"],
            "elbow_values": result["elbow_values"],
            "evaluation_table": result["evaluation_table"],
            "pca_explained_variance": result["pca_explained_variance"],
        }

        return {
            "status": "success",
            "message": "Employee segmentation completed successfully.",
            "best_k": result["best_k"],
            "features_used": result["features_used"],
            "silhouette_scores": result["silhouette_scores"],
            "elbow_values": result["elbow_values"],
            "evaluation_table": result["evaluation_table"],
            "pca_explained_variance": result["pca_explained_variance"],
            "cluster_labels": result["cluster_labels"],
            "cluster_profiles": result["cluster_profiles"],
            "summary": result["summary"],
            "output_path": result["output_path"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while running segmentation: {str(e)}"
        )


@app.post("/upload-prepared-data-to-huggingface")
def upload_prepared_data_to_huggingface():
    files_to_upload = {
        "cleaned_data.csv": CURRENT_DATA.get("cleaned_path"),
        "encoded_data.csv": CURRENT_DATA.get("encoded_path"),
        "scaled_data.csv": CURRENT_DATA.get("scaled_path"),
    }

    if CURRENT_DATA.get("clustered_path") is not None:
        files_to_upload["clustered_data.csv"] = CURRENT_DATA["clustered_path"]

    files_to_upload = {
        hf_name: local_path
        for hf_name, local_path in files_to_upload.items()
        if local_path is not None and os.path.exists(local_path)
    }

    if not files_to_upload:
        raise HTTPException(
            status_code=400,
            detail="No prepared data available. Please upload or load a dataset first."
        )

    uploaded_files = {}

    try:
        for hf_name, local_path in files_to_upload.items():
            hf_url = upload_file_to_huggingface(
                local_path,
                f"prepared_data/{hf_name}"
            )
            uploaded_files[hf_name] = hf_url

        return {
            "status": "success",
            "message": "Prepared data uploaded to Hugging Face successfully.",
            "uploaded_files": uploaded_files,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not upload prepared data: {str(e)}"
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
        "encoded_path": CURRENT_DATA["encoded_path"],
        "scaled_path": CURRENT_DATA["scaled_path"],
        "clustered_path": CURRENT_DATA["clustered_path"],
        "segmentation_summary": CURRENT_DATA["segmentation_summary"],
        "segmentation_metrics": CURRENT_DATA["segmentation_metrics"],
        "risk_path": CURRENT_DATA["risk_path"],
    }


@app.post("/run-risk-detection")
def risk_detection_api():
    try:
        risk_df, summary = run_risk_detection()

        CURRENT_DATA["risk_path"] = summary["output_path"]

        return {
            "status": "success",
            "message": "Risk detection and burnout intelligence completed successfully.",
            "summary": summary,
            "columns_added": [
                "attrition_score",
                "burnout_score",
                "risk_category",
                "burnout_category",
                "anomaly_status",
                "danger_explanation"
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while running risk detection: {str(e)}"
        )


@app.post("/run-recommendations")
def recommendations_api():
    """
    ALA'S RECOMMENDATION SYSTEM (V5):
    Calculates weighted priority scores and generates HR intervention actions 
    based on risk and burnout signals.
    """
    try:
        # Call the logic. Note: The function will automatically load V5 from BASE_PATH
        result_df = generate_recommendations()

        # Convert the DataFrame to a list of dictionaries so Swagger can display it
        output_data = result_df.to_dict(orient="records")

        return {
            "status": "success",
            "version": "5.0",
            "total_records": len(result_df),
            "recommendations": output_data
        }
    except FileNotFoundError as e:
        # Specific error if V5_risk_scored_dataset.csv is missing
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # General server error handling
        raise HTTPException(status_code=500, detail=f"Recommendation Engine Error: {str(e)}")
    