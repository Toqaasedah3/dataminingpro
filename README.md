# WorkPulse Platform

A comprehensive Python-based Data Mining platform for HR analytics, recruitment intelligence, workforce segmentation, and team-ready dataset sharing.

This system enables HR professionals, recruiters, analysts, and data science teams to:

- Upload HR datasets dynamically
- Automatically detect HR-related columns
- Clean and preprocess raw employee data
- Generate multiple prepared dataset versions
- Run employee segmentation using K-Means clustering
- Perform PCA dimensionality reduction
- Visualize HR insights interactively
- Share prepared datasets through Hugging Face Hub
- Build a scalable recruitment intelligence workflow

---

# Table of Contents

- Project Overview
- Key Features
- System Architecture
- Technology Stack
- Dataset Management
- Hugging Face Integration
- Installation Guide
- Project Structure
- Environment Configuration
- Running the Application
- Frontend Dashboard Pages
- API Documentation
- Machine Learning Workflow
- Employee Segmentation Workflow
- Dataset Versioning System
- Team Collaboration Workflow
- Sample Dataset
- Troubleshooting
- Future Enhancements
- Contributing
- License
- Support

---

# Project Overview

The WorkPulse Platform is designed to modernize HR data analysis by automating the full pipeline from raw dataset ingestion to advanced analytics.

It supports:

## HR Analytics
- Employee satisfaction analysis
- Attrition indicators
- Salary analysis
- Department insights
- Promotion trends
- Recruitment intelligence

## Data Science Tasks
- Data preprocessing
- Feature engineering
- Dataset encoding
- Dataset scaling
- Clustering
- PCA visualization

## Team Operations
- Shared prepared datasets
- Hugging Face repository hosting
- Version-controlled data pipelines
- Collaborative ML-ready datasets

---

# Key Features

## 1. Dynamic CSV Upload
- Upload any HR dataset
- Supports flexible column structures
- Automatic validation

## 2. Dataset Preview
- Displays uploaded data
- Missing value detection
- Duplicate detection
- Column summary

## 3. Auto HR Column Detection

Automatically identifies:

- Salary
- Satisfaction level
- Last evaluation
- Number of projects
- Average monthly hours
- Time spent at company
- Work accident
- Promotion
- Department
- Attrition/left status

---

## 4. Data Cleaning & Preprocessing

Includes:

- Missing value handling
- Duplicate removal
- Data type correction
- Categorical normalization
- Label encoding
- Feature scaling

---

## 5. Dataset Versioning

### V0
Raw uploaded dataset

### V1
Cleaned dataset

### V2
Encoded dataset

### V3
Scaled dataset

### V4
Clustered dataset

---

## 6. Employee Segmentation

Uses:
- K-Means clustering
- Silhouette Score optimization
- Automatic best-K selection

---

## 7. PCA Visualization

- 2D dimensionality reduction
- Cluster plotting
- Recruitment segmentation insights

---

## 8. FastAPI Backend

- REST API endpoints
- Dataset processing
- Segmentation pipeline
- Hugging Face integration

---

## 9. Streamlit Frontend

Interactive pages:
- Upload
- Preview
- Cleaning
- Segmentation
- Dataset versions
- Analytics dashboard

---

## 10. Hugging Face Integration

- Upload prepared datasets
- Team sharing
- External ML pipeline support
- Dataset reproducibility

---

# System Architecture

```text
User Upload CSV
      ↓
Frontend (Streamlit Dashboard)
      ↓
Backend API (FastAPI)
      ↓
Column Detection Module
      ↓
Preprocessing Module
      ↓
Dataset Versioning
      ↓
Segmentation Module (K-Means + PCA)
      ↓
Visualization Dashboard
      ↓
Hugging Face Dataset Repository
```

---

# Technology Stack

## Backend
- FastAPI
- Python
- Pandas
- NumPy
- Scikit-learn

## Frontend
- Streamlit
- Plotly

## Machine Learning
- K-Means Clustering
- PCA
- Silhouette Analysis

## Dataset Hosting
- Hugging Face Hub

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/Toqaasedah3/dataminingpro.git
cd dataminingpro
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

If requirements.txt is unavailable:

```bash
pip install fastapi uvicorn streamlit pandas numpy scikit-learn plotly datasets huggingface_hub python-dotenv python-multipart
```

---

# Environment Configuration

Create a `.env` file in the root project directory:

```env
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=toqa66/hr-talent-mining-dataset
```

---

# Project Structure

```text
project-root/
│
├── backend/
│   ├── main.py
│   ├── modules/
│   │   ├── preprocessing.py
│   │   ├── segmentation.py
│   │   ├── column_detection.py
│   │   ├── recommendation_engine.py
│   │   ├── recommendation_engine_cleaned.py
│   │   ├── risk_detection.py
│   │   └── huggingface_utils.py
│   │
│   └── data/
│       └── processed/
│
├── frontend/
│   ├── app.py
│   └── pages/
│
├── requirements.txt
├── README.md
└── .env
```

---

# Running the Application

## 1. Run Backend API

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 2. Run Frontend Dashboard

Open another terminal and activate the virtual environment again:

### Windows

```bash
.\.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Then run:

```bash
streamlit run frontend/app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

# Frontend Dashboard Pages

## Upload Page
- Upload datasets
- Preview uploaded data
- Validate files

## Cleaning Dashboard
- Missing values
- Duplicate analysis
- Dataset transformations

## Segmentation Dashboard
- K-Means clustering
- PCA visualization
- Cluster summaries

## Analytics Dashboard
- HR statistics
- Employee insights
- Department metrics

## Dataset Versions
- Download processed datasets
- Compare versions

---

# API Documentation

## Dataset Endpoints

### Upload Dataset

```http
POST /upload-dataset
```

### Load Raw Dataset from Hugging Face

```http
GET /load-raw-from-huggingface
```

### Load Cleaned Dataset

```http
GET /load-cleaned-from-huggingface
```

### Load Clustered Dataset

```http
GET /load-clustered-from-huggingface
```

### Dataset Information

```http
GET /dataset-info
```

---

## Segmentation Endpoints

### Run Segmentation

```http
POST /run-segmentation
```

---

## Risk Detection Endpoints

### Run Risk Detection

```http
POST /run-risk-detection
```

---

## Recommendation Endpoints

### Run Recommendations

```http
POST /run-recommendations
```

### Run Cleaned Recommendations

```http
POST /run-cleaned-recommendations
```

---

# Machine Learning Workflow

```text
Raw Dataset
    ↓
Cleaning
    ↓
Encoding
    ↓
Scaling
    ↓
Feature Selection
    ↓
K-Means Clustering
    ↓
PCA Reduction
    ↓
Visualization
```

---

# Employee Segmentation Workflow

## Step 1
Load and preprocess HR data.

## Step 2
Scale numerical features.

## Step 3
Apply K-Means clustering.

## Step 4
Determine optimal K using Silhouette Score.

## Step 5
Generate employee clusters.

## Step 6
Reduce dimensions using PCA.

## Step 7
Visualize employee groups.

---

# Dataset Versioning System

| Version | Description |
|---|---|
| V0 | Raw uploaded dataset |
| V1 | Cleaned dataset |
| V2 | Encoded dataset |
| V3 | Scaled dataset |
| V4 | Clustered dataset |

---

# Team Collaboration Workflow

```text
Upload Dataset
      ↓
Generate Prepared Versions
      ↓
Upload to Hugging Face
      ↓
Team Access Shared Data
      ↓
Continue ML Experiments
```

---

# Hugging Face Integration

The system supports:
- Uploading processed datasets
- Downloading prepared datasets
- Team-wide dataset sharing
- Version-controlled data workflows

Processed datasets are automatically saved locally inside:

```text
backend/data/processed
```

---

# Sample Dataset

The project supports HR datasets containing:
- Employee satisfaction
- Salary
- Department
- Evaluation metrics
- Promotion history
- Attrition labels
- Working hours
- Project count

---

# Troubleshooting

## Hugging Face Repository Not Found

If you see:

```text
dataset couldn't be found on the Hugging Face Hub
```

Check:
- Repository name is correct
- Dataset repository is public
- HF_TOKEN is valid
- Dataset file path exists

---

## Backend Not Running

Make sure backend is running:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

---

## Frontend Cannot Connect to Backend

Check API URL:

```python
API_URL = "http://127.0.0.1:8000"
```

---

## Missing Packages

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Future Enhancements

- Deep Learning integration
- Predictive attrition modeling
- Employee recommendation engine
- Real-time analytics
- Dashboard authentication
- Docker deployment
- Cloud hosting
- Multi-company support

---

# Contributing

Contributions are welcome.

Steps:
1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push updates
5. Create pull request

---

# License

This project is intended for educational and research purposes.

---

# Support

For technical support or collaboration:

- GitHub Issues
- Hugging Face Repository
- Team Communication Channels

---

# Final Notes

WorkPulse Platform provides a scalable HR analytics pipeline combining:
- Data Engineering
- Machine Learning
- HR Intelligence
- Team Collaboration
- Dataset Reproducibility

The platform is designed to move beyond traditional academic projects into a production-style intelligent HR analytics system.
