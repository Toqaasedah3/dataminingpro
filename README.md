# HR Talent Mining & Recruitment Intelligence Platform

A practical Python-based Data Mining application for HR analytics and talent recruitment intelligence. This platform allows HR professionals to upload employee datasets, automatically detect HR-related columns, clean and preprocess data, perform employee segmentation using K-Means clustering and PCA dimensionality reduction, and visualize results in an interactive dashboard.

## Features

- **Dynamic CSV Upload**: Upload any HR dataset in CSV format
- **Dataset Preview & Validation**: View and validate uploaded data
- **Auto HR Column Detection**: Automatically identifies common HR columns (salary, satisfaction, department, etc.)
- **Data Cleaning & Preprocessing**: Handles missing values, duplicates, and data type conversions
- **Dataset Versioning**: Saves datasets in three versions:
  - V0: Raw uploaded data
  - V1: Cleaned and preprocessed data
  - V2: Clustered data with segmentation results
- **Employee Segmentation**: Uses K-Means clustering with optimal K selection via silhouette score
- **Dimensionality Reduction**: PCA for 2D visualization of employee clusters
- **Interactive Dashboard**: Streamlit-based frontend with Plotly charts
- **API Backend**: FastAPI-powered REST API for data processing
- **Hugging Face Integration**: Optional dataset versioning on Hugging Face Hub

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hr-talent-mining-project.git
cd hr-talent-mining-project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**On Windows:**
```bash
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Setup

### 1. Project Structure

Ensure your project folder has the following structure:

```
hr-talent-mining-project/
├── backend/
│   ├── main.py
│   └── modules/
│       ├── column_detection.py
│       ├── preprocessing.py
│       ├── segmentation.py
│       └── huggingface_utils.py
├── frontend/
│   └── app.py
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── requirements.txt
├── README.md
└── .env.example
```

### 2. Environment Configuration (Optional)

If you want to enable Hugging Face dataset versioning:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Hugging Face credentials:
   ```
   HF_TOKEN=your_huggingface_token_here
   HF_REPO_ID=your-username/hr-talent-mining-dataset
   ```

   - Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a dataset repository on Hugging Face Hub

## Running the Application

### 1. Start the Backend API

Open a terminal and run:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`

You can view the API documentation at: `http://127.0.0.1:8000/docs`

### 2. Start the Frontend Dashboard

Open another terminal and run:

```bash
streamlit run frontend/app.py
```

The dashboard will be available at: `http://localhost:8501`

## Usage

1. **Upload Dataset**: Use the "Upload Dataset" page to upload a CSV file containing HR data.

2. **Data Preview**: Review the uploaded data, check for missing values, and see detected columns.

3. **Preprocessing**: Click "Send to Backend and Clean Dataset" to process the data.

4. **Employee Segmentation**: Go to "Employee Segmentation" page and click "Run Segmentation" to perform K-Means clustering with PCA.

5. **View Results**: Explore cluster summaries, PCA visualizations, and download processed datasets.

6. **Dataset Versions**: Check the "Dataset Versions" page to see saved file paths and API status.

## Sample Dataset

A sample HR dataset is provided at `data/raw/sample_hr_dataset.csv` for testing the application.

## API Endpoints

- `GET /`: Health check
- `POST /upload-dataset`: Upload and process CSV file
- `POST /run-segmentation`: Perform employee segmentation
- `GET /dataset-info`: Get current dataset information

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and virtual environment is activated.

2. **Port Conflicts**: If ports 8000 or 8501 are busy, change them in the run commands.

3. **Hugging Face Upload Fails**: Check your `.env` file and Hugging Face token permissions.

4. **Segmentation Errors**: Ensure your dataset has at least 2 numeric columns for clustering.

### Logs

- Backend logs appear in the terminal running uvicorn
- Frontend logs appear in the terminal running streamlit

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.
