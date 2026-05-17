import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


PROCESSED_DIR = os.path.join("backend", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]


PREFERRED_FEATURES = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "TotalWorkingYears",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "JobLevel",
    "DistanceFromHome",
    "PerformanceRating"
]


def select_best_features(df: pd.DataFrame):
    df = df.copy()

    df = df.drop(
        columns=[col for col in DROP_COLUMNS if col in df.columns],
        errors="ignore"
    )

    available_features = [
        col for col in PREFERRED_FEATURES
        if col in df.columns
    ]

    if available_features:
        feature_df = df[available_features].copy()
    else:
        feature_df = df.copy()

    for col in feature_df.columns:
        if feature_df[col].dtype == "object":
            mode_value = feature_df[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            feature_df[col] = feature_df[col].fillna(fill_value)
        else:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            median_value = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(
                median_value if pd.notna(median_value) else 0
            )

    feature_df = pd.get_dummies(feature_df, drop_first=True)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(0)

    nunique = feature_df.nunique()
    feature_df = feature_df[nunique[nunique > 1].index]

    if feature_df.empty:
        raise ValueError("No useful features found for segmentation.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_df)

    return scaled_data, feature_df.columns.tolist()


def apply_pca_svd(scaled_data, n_components=2):
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
        random_state=42
    )

    pca_result = pca.fit_transform(scaled_data)

    explained_variance = [
        float(value) for value in pca.explained_variance_ratio_
    ]

    return pca_result, explained_variance


def find_best_k(data, min_k=2, max_k=10):
    results = []

    max_k = min(max_k, len(data) - 1)

    if max_k < min_k:
        raise ValueError("Dataset is too small for clustering.")

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=30,
            max_iter=500
        )

        labels = model.fit_predict(data)

        results.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(data, labels)),
            "davies_bouldin": float(davies_bouldin_score(data, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(data, labels))
        })

    best_result = max(results, key=lambda x: x["silhouette"])
    best_k = best_result["k"]

    silhouette_scores = {
        item["k"]: item["silhouette"]
        for item in results
    }

    elbow_values = {
        item["k"]: item["inertia"]
        for item in results
    }

    evaluation_table = results

    return best_k, silhouette_scores, elbow_values, evaluation_table


def build_cluster_label(cluster_profile, overall_profile):
    labels = []

    def above(col):
        return col in cluster_profile and cluster_profile[col] > overall_profile[col]

    def below(col):
        return col in cluster_profile and cluster_profile[col] < overall_profile[col]

    if above("MonthlyIncome"):
        labels.append("High Income")
    elif below("MonthlyIncome"):
        labels.append("Low Income")

    if below("JobSatisfaction"):
        labels.append("Low Satisfaction")
    elif above("JobSatisfaction"):
        labels.append("Satisfied")

    if below("WorkLifeBalance"):
        labels.append("Poor Work-Life Balance")
    elif above("WorkLifeBalance"):
        labels.append("Good Work-Life Balance")

    if above("YearsAtCompany"):
        labels.append("Long Tenure")
    elif below("YearsAtCompany"):
        labels.append("New Employees")

    if above("PerformanceRating"):
        labels.append("High Performance")

    if above("DistanceFromHome"):
        labels.append("Long Commute")

    if not labels:
        return "General Employee Segment"

    return " / ".join(labels[:3])


def label_clusters(result_df: pd.DataFrame):
    labels = {}

    numeric_cols = result_df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = [
        col for col in numeric_cols
        if col not in ["cluster", "pca_1", "pca_2"]
    ]

    overall_profile = result_df[numeric_cols].mean()

    for cluster_id in sorted(result_df["cluster"].unique()):
        cluster_data = result_df[result_df["cluster"] == cluster_id]
        cluster_profile = cluster_data[numeric_cols].mean()

        labels[int(cluster_id)] = build_cluster_label(
            cluster_profile,
            overall_profile
        )

    return labels


def build_cluster_profiles(result_df: pd.DataFrame):
    profile_columns = [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "PerformanceRating"
    ]

    available_cols = [
        col for col in profile_columns
        if col in result_df.columns
    ]

    if not available_cols:
        return []

    profiles = (
        result_df
        .groupby("cluster")[available_cols]
        .mean()
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    return profiles


def run_segmentation(original_df: pd.DataFrame = None):
    if original_df is None or original_df.empty:
        raise ValueError("Original uploaded company dataset is required.")

    scaled_data, features_used = select_best_features(original_df)

    pca_result, pca_explained_variance = apply_pca_svd(
        scaled_data,
        n_components=2
    )

    best_k, silhouette_scores, elbow_values, evaluation_table = find_best_k(
        scaled_data,
        min_k=2,
        max_k=10
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=50,
        max_iter=700
    )

    clusters = final_model.fit_predict(scaled_data)

    result_df = original_df.copy()
    result_df["cluster"] = clusters
    result_df["pca_1"] = pca_result[:, 0]
    result_df["pca_2"] = pca_result[:, 1]

    cluster_labels = label_clusters(result_df)
    result_df["cluster_label"] = result_df["cluster"].map(cluster_labels)

    summary_df = (
        result_df
        .groupby(["cluster", "cluster_label"])
        .size()
        .reset_index(name="employee_count")
    )

    cluster_profiles = build_cluster_profiles(result_df)

    output_path = os.path.join(PROCESSED_DIR, "clustered_data.csv")
    result_df.to_csv(output_path, index=False)

    return {
        "best_k": int(best_k),
        "features_used": features_used,
        "silhouette_scores": silhouette_scores,
        "elbow_values": elbow_values,
        "evaluation_table": evaluation_table,
        "pca_explained_variance": pca_explained_variance,
        "cluster_labels": cluster_labels,
        "cluster_profiles": cluster_profiles,
        "summary": summary_df.to_dict(orient="records"),
        "clustered_df": result_df,
        "output_path": output_path,
    }