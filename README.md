# 🚀 NASA NEO Interactive Universe & Hazard Classification

<p align="center">

  <!-- GitHub Actions -->
  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_pusher.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_pusher.yml?branch=main&label=Data%20Pusher&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/predict.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/predict.yml?branch=main&label=Predict%20Fresh%20Data&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_drift_check.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_drift_check.yml?branch=main&label=Data%20Drift%20Check&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/continous_integration.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/continous_integration.yml?branch=main&label=Continuous%20Integration&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/auto_risk_analysis.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/auto_risk_analysis.yml?branch=main&label=Auto%20Risk%20Analysis&logo=github" />
  </a>

  <!-- MLflow -->
  <a href="https://dagshub.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection.mlflow/#/experiments/8">
    <img src="https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow" />
  </a>

  <!-- PostgreSQL -->
  <img src="https://img.shields.io/badge/Database-PostgreSQL-blue?logo=postgresql" />

  <!-- FastAPI -->
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />

  <!-- Three.js -->
  <img src="https://img.shields.io/badge/3D-Three.js-black?logo=three.js&logoColor=white" />

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?logo=python" />

  <!-- License -->
  <img src="https://img.shields.io/badge/License-NonCommercial-orange" />

</p>

This project is a **high-performance, production-grade 3D web application** that visualizes **32,001 NASA Near-Earth Objects (NEOs)** and classifies them as *hazardous* or *non-hazardous*. It combines a sophisticated **Three.js 3D simulation** with a robust **MLOps pipeline** automated via GitHub Actions, DVC, and FastAPI.

---

## 📑 Project Overview

- **Interactive Universe**: A cinematic 3D simulation (Orrery) of the solar system featuring 32,000+ real-time tracked asteroids.
- **AI-Driven Prediction**: Real-time hazard classification using ML models trained with NASA orbital data.
- **3D Scale Comparator**: High-fidelity visual tool to compare asteroid sizes against Earth in real-time.
- **Full-Stack Architecture**: Modern split-module design with a specialized FastAPI backend and a glassmorphic Vanilla JS frontend.
- **End-to-End MLOps**: Automated data ingestion, preprocessing (SMOTE), drift detection, and monitoring via MLflow and Grafana.

---

## ⚙️ Tech Stack

### Frontend (Interactive UI)
- **3D Engine**: Three.js (WebGL)
- **Styling**: Vanilla CSS with Glassmorphism
- **State Management**: Asynchronous JS with custom state-driven components

### Backend (Production API)
- **Framework**: FastAPI (Asynchronous)
- **Database**: Neon (Serverless PostgreSQL)
- **Real-time**: WebSockets for live asteroid tracking
- **Data Versioning**: DVC (Data Version Control)

### Machine Learning & MLOps
- **ML Engine**: Scikit-Learn, XGBoost, CatBoost
- **Pipeline**: GitHub Actions (Scheduled Workflows)
- **Experiment Tracking**: MLflow (hosted on DAGsHub)
- **Monitoring**: Grafana Dashboards

---

## 🌌 Interactive Modules

### 1. High-Fidelity 3D Orrery
- Real-time visualization of 32,001 unique NASA asteroids.
- Cinematic camera controls with zoom-to-asteroid functionality.
- Procedural textures for celestial bodies and starfield environments.

### 2. Asteroid-to-Earth Comparator
- Dynamic 3D model scaling based on real physical diameters.
- Synchronized zoom effects for precise scale perception.
- Targeting reticle and AI-driven risk indicators.

### 3. AI Explorer & Leaderboard
- Name-based asteroid search and advanced filtering.
- AI risk assessment scoring displayed in real-time.
- WebSocket-powered "Recent Discoveries" live feed.

### 4. Technical Dashboard (MLOps)
- Visual documentation of the end-to-end data pipeline.
- DVC-driven data flow visualization.
- Real-time backend health monitoring.

---

## 🌳 Repository Structure

```
├── 📁 .github/             # GitHub Actions Workflows (CI/CD, Ingestion, Drift)
├── 📁 backend/             # FastAPI Application (API v2.0)
│   ├── 🐍 main.py          # API Entry Point
│   ├── 🐍 models.py        # SQLAlchemy/Neon Models
│   ├── 🐍 database.py      # NeonDB Connection Pool & Watcher
├── 📁 frontend/            # High-Fidelity Web Interface
│   ├── 📁 src/             # Specialized 3D & UI Components
│   │   ├── 💎 orrery.js    # 3D Solar System Engine
│   │   ├── 💎 comparator.js # 3D Scale Comparison
│   │   ├── 📂 mlops.js     # Technical Dashboard
├── 📁 src/                 # ML Pipeline Source Code (V1)
│   ├── 📁 custom/          # Data Transformation & Model Training
├── 📁 Data/                # Local data storage (DVC tracked)
├── 📁 Notebook/            # Research & Exploratory Analysis
├── 🐍 app.py               # Legacy Flask Gateway
├── 🧊 dvc.yaml             # Data Pipeline Orchestration
└── 📖 README.md            # This documentation
```

---

## 📊 Monitoring & Alerts

- **Grafana Dashboards**: Publicly accessible dashboards tracking ingestion, performance, and drift.
- **Data Drift Detection**: Automated weekly checks with email notifications via GitHub Actions.
- **Real-time Watcher**: Background service polling for new unique asteroid discoveries.

---

## 👨‍💻 Author

**Subrat Mishra**

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue?style=flat&logo=internet-explorer)](https://mishra-subrat.framer.website)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/subrat1920/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Subrat1920)
[![Medium](https://img.shields.io/badge/Medium-Read-black?style=flat&logo=medium)](https://medium.com/@subrat1920/1671404ef449)
