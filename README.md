# mlops-model-deployment
Complete MLOps workflow demonstrating data versioning, model training, experiment tracking, API deployment, CI/CD integration, and monitoring


## 📌 Project Title
**End-to-End MLOps Pipeline: From Model Development to Deployment and Monitoring**

## 📄 Description
This project demonstrates a full MLOps workflow using the California Housing dataset. It includes data preprocessing, training models (Linear Regression & Decision Tree), tracking experiments with MLflow, serving predictions through a FastAPI-based REST API, Docker containerization, CI/CD using GitHub Actions, and logging incoming prediction requests.

---

## 📁 Project Structure
```
project/
├── .github/workflows/ci.yml         # CI/CD pipeline with GitHub Actions
├── api/app.py                       # FastAPI app for predictions
├── data/                            # Raw and processed data
│   └── raw/housing.csv
│   └── processed/housing_cleaned.csv
├── logs/predictions.log             # Log of prediction requests
├── models/best_model.pkl            # Trained and saved model
├── notebooks/eda.ipynb              # EDA and visualization
├── src/                             # Core scripts
│   ├── data_preprocessing.py        # Preprocessing logic
│   ├── train.py                     # Model training & MLflow tracking
│   ├── evaluate.py                  # (Optional) Evaluation metrics
│   └── predict.py                   # Prediction helpers
├── tests/test_api.py                # Unit test for API
├── Dockerfile                       # Docker config for containerization
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files/folders to ignore in git
└── deploy.sh                        # Shell script for deployment
```

---

## ⚙️ Features Implemented
- Data cleaning and scaling using `sklearn`
- Training multiple models and tracking them using `MLflow`
- Registering and saving the best-performing model
- REST API for predictions using `FastAPI`
- Dockerized API service
- Automated testing and linting with GitHub Actions
- CI/CD pipeline that builds and runs the container
- Logging prediction requests to a file

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/mlops-housing-regression-pipeline.git
cd mlops-housing-regression-pipeline
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train model
```bash
python src/train.py
```

### Run API locally
```bash
uvicorn api.app:app --reload
```

### Test API
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"features": [8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]}'
```

---

## 🐳 Docker
```bash
docker build -t mlops-api .
docker run -p 8000:8000 mlops-api
```

---

## 🔁 CI/CD with GitHub Actions
- Code is automatically linted and tested on every push.
- Docker image is built and ready to be pushed/deployed.

---

## 📊 Logging and Monitoring
- Logs all API requests and responses to `logs/predictions.log`
- (Optional) Expose `/metrics` endpoint using Prometheus if needed.

---

## 📦 Dataset
California Housing dataset: available via `sklearn.datasets.fetch_california_housing()` or `housing.csv` in `data/raw/`

---

## 👨‍💻 Author
**Your Name** (Add your GitHub or email)

---

## 📄 License
[MIT License](LICENSE) (Optional)

