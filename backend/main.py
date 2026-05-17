import os
import pandas as pd

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


PROCESSED_DIR = os.path.join("backend", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


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

    "clustered_df": None,
    "clustered_path": None,
    "segmentation_summary": None,
    "segmentation_metrics": None,

    "risk_df": None,
    "risk_path": None,

    "recommendation_df": None,
    "recommendation_path": None,
}


@app.get("/")
def root():
    return {"message": "HR Talent Mining API is running."}


def reset_analysis_results():
    CURRENT_DATA.update({
        "clustered_df": None,
        "clustered_path": None,
        "segmentation_summary": None,
        "segmentation_metrics": None,
        "risk_df": None,
        "risk_path": None,
        "recommendation_df": None,
        "recommendation_path": None,
    })


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
    })

    reset_analysis_results()
    return clean_result


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        clean_result = process_dataset(df)

        return {
            "status": "success",
            "source": "uploaded_company_csv",
            "message": "Company dataset uploaded, cleaned, encoded, and scaled successfully.",
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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while uploading/cleaning dataset: {str(e)}"
        )


@app.post("/run-segmentation")
def segmentation_api():
    if CURRENT_DATA["cleaned_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No uploaded dataset found. Please upload a company CSV first."
        )

    try:
        result = run_segmentation(
            original_df=CURRENT_DATA["cleaned_df"]
        )

        CURRENT_DATA["clustered_path"] = result["output_path"]
        CURRENT_DATA["clustered_df"] = pd.read_csv(result["output_path"])
        CURRENT_DATA["segmentation_summary"] = result["summary"]
        CURRENT_DATA["segmentation_metrics"] = {
            "best_k": result["best_k"],
            "features_used": result["features_used"],
            "silhouette_scores": result["silhouette_scores"],
            "elbow_values": result["elbow_values"],
            "evaluation_table": result["evaluation_table"],
            "pca_explained_variance": result["pca_explained_variance"],
        }

        CURRENT_DATA["risk_df"] = None
        CURRENT_DATA["risk_path"] = None
        CURRENT_DATA["recommendation_df"] = None
        CURRENT_DATA["recommendation_path"] = None

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


@app.post("/run-risk-detection")
def risk_detection_api():
    try:
        if CURRENT_DATA["clustered_df"] is None:
            if CURRENT_DATA["cleaned_df"] is None:
                raise HTTPException(
                    status_code=400,
                    detail="No uploaded dataset found. Please upload a company CSV first."
                )

            seg_result = run_segmentation(
                original_df=CURRENT_DATA["cleaned_df"]
            )

            CURRENT_DATA["clustered_path"] = seg_result["output_path"]
            CURRENT_DATA["clustered_df"] = pd.read_csv(seg_result["output_path"])
            CURRENT_DATA["segmentation_summary"] = seg_result["summary"]
            CURRENT_DATA["segmentation_metrics"] = {
                "best_k": seg_result["best_k"],
                "features_used": seg_result["features_used"],
                "silhouette_scores": seg_result["silhouette_scores"],
                "elbow_values": seg_result["elbow_values"],
                "evaluation_table": seg_result["evaluation_table"],
                "pca_explained_variance": seg_result["pca_explained_variance"],
            }

        risk_df, summary = run_risk_detection(
            df=CURRENT_DATA["clustered_df"]
        )

        CURRENT_DATA["risk_df"] = risk_df
        CURRENT_DATA["risk_path"] = summary["output_path"]
        CURRENT_DATA["recommendation_df"] = None
        CURRENT_DATA["recommendation_path"] = None

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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while running risk detection: {str(e)}"
        )


@app.post("/run-recommendations")
def recommendations_api():
    try:
        if CURRENT_DATA["risk_df"] is None:
            if CURRENT_DATA["clustered_df"] is None:
                raise HTTPException(
                    status_code=400,
                    detail="Please run segmentation and risk detection first, or run the full pipeline."
                )

            risk_df, risk_summary = run_risk_detection(
                df=CURRENT_DATA["clustered_df"]
            )

            CURRENT_DATA["risk_df"] = risk_df
            CURRENT_DATA["risk_path"] = risk_summary["output_path"]

        result_df = generate_recommendations(
            df=CURRENT_DATA["risk_df"]
        )

        recommendation_path = os.path.join(
            PROCESSED_DIR,
            "V6_recommendations.csv"
        )

        result_df.to_csv(recommendation_path, index=False)

        CURRENT_DATA["recommendation_df"] = result_df
        CURRENT_DATA["recommendation_path"] = recommendation_path

        return {
            "status": "success",
            "version": "6.0",
            "message": "Recommendations generated successfully.",
            "total_records": int(len(result_df)),
            "output_path": recommendation_path,
            "recommendations": result_df.to_dict(orient="records"),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation Engine Error: {str(e)}"
        )


@app.post("/run-full-pipeline")
def run_full_pipeline():
    pipeline_status = {
        "upload_cleaning": {"status": "pending", "message": ""},
        "segmentation": {"status": "pending", "message": ""},
        "risk_detection": {"status": "pending", "message": ""},
        "recommendations": {"status": "pending", "message": ""},
    }

    if CURRENT_DATA["cleaned_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No uploaded dataset found. Please upload a company CSV first."
        )

    pipeline_status["upload_cleaning"] = {
        "status": "completed",
        "message": "Uploaded company dataset is already cleaned and ready.",
        "rows": int(CURRENT_DATA["cleaned_df"].shape[0]),
        "columns": int(CURRENT_DATA["cleaned_df"].shape[1]),
    }

    try:
        seg_result = run_segmentation(
            original_df=CURRENT_DATA["cleaned_df"]
        )

        CURRENT_DATA["clustered_path"] = seg_result["output_path"]
        CURRENT_DATA["clustered_df"] = pd.read_csv(seg_result["output_path"])
        CURRENT_DATA["segmentation_summary"] = seg_result["summary"]
        CURRENT_DATA["segmentation_metrics"] = {
            "best_k": seg_result["best_k"],
            "features_used": seg_result["features_used"],
            "silhouette_scores": seg_result["silhouette_scores"],
            "elbow_values": seg_result["elbow_values"],
            "evaluation_table": seg_result["evaluation_table"],
            "pca_explained_variance": seg_result["pca_explained_variance"],
        }

        pipeline_status["segmentation"] = {
            "status": "completed",
            "message": f"Segmentation completed successfully. Best K = {seg_result['best_k']}.",
            "best_k": seg_result["best_k"],
            "output_path": seg_result["output_path"],
        }

    except Exception as e:
        pipeline_status["segmentation"] = {
            "status": "failed",
            "message": str(e)
        }
        return {
            "status": "failed",
            "failed_at": "segmentation",
            "pipeline_status": pipeline_status
        }

    try:
        risk_df, risk_summary = run_risk_detection(
            df=CURRENT_DATA["clustered_df"]
        )

        CURRENT_DATA["risk_df"] = risk_df
        CURRENT_DATA["risk_path"] = risk_summary["output_path"]

        pipeline_status["risk_detection"] = {
            "status": "completed",
            "message": "Risk detection completed successfully.",
            "summary": risk_summary,
        }

    except Exception as e:
        pipeline_status["risk_detection"] = {
            "status": "failed",
            "message": str(e)
        }
        return {
            "status": "failed",
            "failed_at": "risk_detection",
            "pipeline_status": pipeline_status
        }

    try:
        result_df = generate_recommendations(
            df=CURRENT_DATA["risk_df"]
        )

        recommendation_path = os.path.join(
            PROCESSED_DIR,
            "V6_recommendations.csv"
        )

        result_df.to_csv(recommendation_path, index=False)

        CURRENT_DATA["recommendation_df"] = result_df
        CURRENT_DATA["recommendation_path"] = recommendation_path

        pipeline_status["recommendations"] = {
            "status": "completed",
            "message": f"Recommendations generated for {len(result_df)} employees.",
            "total_records": int(len(result_df)),
            "output_path": recommendation_path,
        }

    except Exception as e:
        pipeline_status["recommendations"] = {
            "status": "failed",
            "message": str(e)
        }
        return {
            "status": "failed",
            "failed_at": "recommendations",
            "pipeline_status": pipeline_status
        }

    return {
        "status": "success",
        "message": "Full pipeline executed successfully using uploaded company dataset.",
        "pipeline_status": pipeline_status,
        "outputs": {
            "cleaned_path": CURRENT_DATA["cleaned_path"],
            "encoded_path": CURRENT_DATA["encoded_path"],
            "scaled_path": CURRENT_DATA["scaled_path"],
            "clustered_path": CURRENT_DATA["clustered_path"],
            "risk_path": CURRENT_DATA["risk_path"],
            "recommendation_path": CURRENT_DATA["recommendation_path"],
        }
    }


@app.post("/upload-prepared-data-to-huggingface")
def upload_prepared_data_to_huggingface():
    files_to_upload = {
        "cleaned_data.csv": CURRENT_DATA.get("cleaned_path"),
        "encoded_data.csv": CURRENT_DATA.get("encoded_path"),
        "scaled_data.csv": CURRENT_DATA.get("scaled_path"),
        "clustered_data.csv": CURRENT_DATA.get("clustered_path"),
        "V5_risk_scored_dataset.csv": CURRENT_DATA.get("risk_path"),
        "V6_recommendations.csv": CURRENT_DATA.get("recommendation_path"),
    }

    files_to_upload = {
        hf_name: local_path
        for hf_name, local_path in files_to_upload.items()
        if local_path is not None and os.path.exists(local_path)
    }

    if not files_to_upload:
        raise HTTPException(
            status_code=400,
            detail="No prepared data available. Please upload a company dataset and run the pipeline first."
        )

    uploaded_files = {}

    try:
        for hf_name, local_path in files_to_upload.items():
            upload_result = upload_file_to_huggingface(
                local_path,
                f"prepared_data/{hf_name}"
            )
            uploaded_files[hf_name] = upload_result

        return {
            "status": "success",
            "message": "Prepared pipeline outputs uploaded to Hugging Face successfully.",
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
            "message": "No company dataset uploaded yet."
        }

    df = CURRENT_DATA["raw_df"]

    return {
        "status": "available",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "column_names": list(df.columns),
        "detected_columns": CURRENT_DATA["detected_columns"],
        "missing_important_columns": missing_important_columns(
            CURRENT_DATA["detected_columns"]
        ),
        "paths": {
            "raw_path": CURRENT_DATA["raw_path"],
            "cleaned_path": CURRENT_DATA["cleaned_path"],
            "encoded_path": CURRENT_DATA["encoded_path"],
            "scaled_path": CURRENT_DATA["scaled_path"],
            "clustered_path": CURRENT_DATA["clustered_path"],
            "risk_path": CURRENT_DATA["risk_path"],
            "recommendation_path": CURRENT_DATA["recommendation_path"],
        },
        "segmentation_summary": CURRENT_DATA["segmentation_summary"],
        "segmentation_metrics": CURRENT_DATA["segmentation_metrics"],
        "has_risk_results": CURRENT_DATA["risk_df"] is not None,
        "has_recommendations": CURRENT_DATA["recommendation_df"] is not None,
    }


@app.get("/dashboard-data")
def dashboard_data():
    if CURRENT_DATA["recommendation_df"] is None:
        raise HTTPException(
            status_code=400,
            detail="No dashboard data available. Please run the full pipeline first."
        )

    recs = CURRENT_DATA["recommendation_df"]

    return {
        "status": "success",
        "total_employees": int(len(recs)),
        "priority_distribution": recs["priority"].value_counts().to_dict()
        if "priority" in recs.columns else {},
        "risk_distribution": recs["risk_category"].value_counts().to_dict()
        if "risk_category" in recs.columns else {},
        "burnout_distribution": recs["burnout_category"].value_counts().to_dict()
        if "burnout_category" in recs.columns else {},
        "recommendations": recs.to_dict(orient="records"),
        "segmentation_summary": CURRENT_DATA["segmentation_summary"],
        "segmentation_metrics": CURRENT_DATA["segmentation_metrics"],
        "paths": {
            "clustered_path": CURRENT_DATA["clustered_path"],
            "risk_path": CURRENT_DATA["risk_path"],
            "recommendation_path": CURRENT_DATA["recommendation_path"],
        }
    }