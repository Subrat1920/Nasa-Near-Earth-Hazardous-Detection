# 🚀 NASA NEO Hazard Classification  

# 🚀 NASA NEO Hazard Classification  

[![Data Pusher](https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_pusher.yml?branch=main&logo=github)](https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_pusher.yml)
[![Predict Fresh Data](https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/predict.yml?branch=main&logo=github)](https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/predict.yml)
[![Data Drift Check](https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_drift_check.yml?branch=main&logo=github)](https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_drift_check.yml)  
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)](https://dagshub.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection)  
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue?logo=postgresql)  
![Grafana](https://img.shields.io/badge/Monitoring-Grafana-orange?logo=grafana)  
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)  
![License](https://img.shields.io/badge/License-MIT-green)  

This project builds a **production-grade ML pipeline** to classify **Near-Earth Objects (NEOs)** as *hazardous* or *non-hazardous*.  
The pipeline automates data ingestion, preprocessing, model training, monitoring, and drift detection using **GitHub Actions, PostgreSQL, MLflow, DAGsHub, and Grafana**.  
)  

This project builds a **production-grade ML pipeline** to classify **Near-Earth Objects (NEOs)** as *hazardous* or *non-hazardous*.  
The pipeline automates data ingestion, preprocessing, model training, monitoring, and drift detection using **GitHub Actions, PostgreSQL, MLflow, DAGsHub, and Grafana**.  

---


## 📑 Project Overview  

- **Goal**: Predict whether an asteroid/NEO poses a potential hazard to Earth.  
- **Data Source**: NASA Near Earth Object (NEO) dataset (orbital data, diameters, velocities, etc.).  
- **Core Features**:  
  - Automated data ingestion (weekly updates)  
  - Preprocessing & handling class imbalance (SMOTE)  
  - Model training with **MLflow tracking on DAGsHub**  
  - Logging models & preprocessors into **PostgreSQL**  
  - **Grafana dashboards** for monitoring & visualization  
  - Weekly **data drift detection** with notifications  

---

## ⚙️ Tech Stack  

- **Orchestration**: GitHub Actions (scheduled workflows)  
- **Database**: PostgreSQL  
- **Machine Learning**: scikit-learn, imbalanced-learn (SMOTE)  
- **Experiment Tracking**: MLflow (integrated with DAGsHub)  
- **Monitoring & Visualization**: Grafana  
- **Alerts**: Email notifications + Grafana alerts  

---

## 📂 Workflow  

### 1. Data Ingestion  
- GitHub Actions scheduled weekly workflow.  
- Fetches latest NASA NEO dataset.  
- Loads data into **PostgreSQL**.  

### 2. Data Transformation  
- Data cleaning, feature scaling, and encoding.  
- **SMOTE** applied to balance hazardous vs non-hazardous classes.  

### 3. Model Training  
- Multiple ML models trained (Logistic Regression, Random Forest, XGBoost, etc.).  
- Tracked with **MLflow + DAGsHub authentication**.  
- Best model automatically selected.  

### 4. Model Logging  
- Best model + preprocessing pipeline stored in **PostgreSQL**.  
- Enables consistent inference and reproducibility.  

### 5. Monitoring with Grafana  
- Grafana dashboard connected to PostgreSQL.  
- Visualizes:  
  - Ingestion stats  
  - Model performance (accuracy, precision, recall, F1)  
  - Drift metrics  

### 6. Data Drift Detection  
- Weekly scheduled GitHub Action.  
- Drift metrics computed & logged in PostgreSQL.  
- Results visualized in Grafana.  

### 7. Notifications  
- **Email alerts** triggered if drift detected.  
- **Grafana notification channels** enabled for anomalies.  

---
## 🌳 Project Tree

```
├── 📁 .github/
│   └── 📁 workflows/
│       ├── ⚙️ data_drift_check.yml
│       ├── ⚙️ data_pusher.yml
│       └── ⚙️ predict.yml
├── 📁 Data/
│   └── 📄 neo_data.csv
├── 📁 pusher/
│   ├── 🐍 __init__.py
│   └── 🐍 weekly_data_pusher.py
├── 📁 src/
│   ├── 📁 constants/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config_entity.py
│   │   ├── 🐍 entity.py
│   │   └── 🐍 params.py
│   ├── 📁 custom/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 data_ingestion.py
│   │   ├── 🐍 data_transformation.py
│   │   ├── 🐍 data_validation.py
│   │   └── 🐍 model_trainer.py
│   ├── 📁 pipeline/
│   │   ├── 📁 prediction/
│   │   │   ├── 🐍 __ini__.py
│   │   │   └── 🐍 predict.py
│   │   └── 🐍 __init__.py
│   ├── 📁 utils/
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 utils.py
│   ├── 🐍 __init__.py
│   ├── 🐍 exception.py
│   └── 🐍 logging.py
├── 📁 tests/
│   └── 🐍 __init__.py
├── 🚫 .gitignore
├── 📖 README.md
├── 🐍 main.py
└── 📄 requirements.txt
```


---


## 📊 Grafana Dashboards  

- **Data Ingestion**: Track weekly data updates.  
- **Model Performance**: Compare metrics over time.  
- **Drift Monitoring**: Detect and visualize changes in data distribution.  

---

## 🔔 Alerts  

- **Email Notification** → Sent when data drift detected.  
- **Grafana Alerts** → Triggered for abnormal performance drops.  

---

## 🚀 Future Improvements  

- Deploy inference API for real-time predictions.  
- Add Docker + Kubernetes for scalable deployment.  
- Integrate SHAP for explainability of hazardous predictions.  

---

## 📌 Author  

**Subrat Mishra**  
[Portfolio](https://mishra-subrat.framer.website) | [LinkedIn](https://www.linkedin.com/in/subrat1920/) | [GitHub](https://github.com/Subrat1920) | [Medium](https://medium.com/@subrat1920)  

---
