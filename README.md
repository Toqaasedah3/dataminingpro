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

The HR Talent Mining Platform is designed to modernize HR data analysis by automating the full pipeline from raw dataset ingestion to advanced analytics.

It supports:

### HR Analytics:
- Employee satisfaction analysis
- Attrition indicators
- Salary analysis
- Department insights
- Promotion trends
- Recruitment intelligence

### Data Science Tasks:
- Data preprocessing
- Feature engineering
- Dataset encoding
- Dataset scaling
- Clustering
- PCA visualization

### Team Operations:
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

## 4. Data Cleaning & Preprocessing
Includes:

- Missing value handling
- Duplicate removal
- Data type correction
- Categorical normalization
- Label encoding
- Feature scaling

## 5. Dataset Versioning
### V0:
Raw uploaded dataset

### V1:
Cleaned dataset

### V2:
Encoded dataset

### V3:
Scaled dataset

### V4:
Clustered dataset

## 6. Employee Segmentation
Uses:
- K-Means clustering
- Silhouette Score optimization
- Automatic best-K selection

## 7. PCA Visualization
- 2D dimensionality reduction
- Cluster plotting
- Recruitment segmentation insights

## 8. FastAPI Backend
- REST API endpoints
- Dataset processing
- Segmentation pipeline
- Hugging Face integration

## 9. Streamlit Frontend
Interactive pages:
- Upload
- Preview
- Cleaning
- Segmentation
- Dataset versions
- Analytics dashboard

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
