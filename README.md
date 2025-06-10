<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology">
  </a>
</p>

<h1 align="center"><b>DEVELOPMENT AND OPERATION OF MACHINE LEARNING SYSTEMS</b></h1>

# MLflow Pipeline for Wine Quality Prediction

- Watch the main flow demo video [here](https://drive.google.com/file/d/1fqHUXSFL31XyWmoZeUkEIUw4oRfo9X3y/view)  
- Watch the docker-compose demo video [here](https://drive.google.com/file/d/109bz4EcAOxRBOUyqWTE7Xr9S-T_qwIRp/view)

## Overview

This project implements an end-to-end machine learning pipeline to predict wine quality ([dataset source](https://www.kaggle.com/datasets/piyushagni5/white-wine-quality)), using Metaflow and MLflow. The pipeline includes the following steps: data loading, exploratory data analysis (EDA), model training, hyperparameter optimization, and model comparison.

## Pipeline Steps

### 1. Load Dataset (`load_dataset`)
- Load wine quality dataset from MLflow's data source (view [here](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv))
- Version control using timestamps
- Log metadata and data statistics to MLflow
- Supports both local and remote (internet) data sources

### 2. Exploratory Data Analysis (EDA) (`eda`)
- Generate detailed dataset statistics
- Create interactive charts using VegaChart:
  - Wine quality distribution
  - Feature-quality relationships
  - Correlation matrix
- All charts are shown in Metaflow cards

### 3. Data Preparation (`load_data`)
- Split data into training and testing sets (80/20)
- Save split data for parallel model training

### 4. Model Training (`train_models`)
- Train multiple classifiers [list here](https://github.com/hyperopt/hyperopt-sklearn#classifiers) in parallel
- Optional hyperparameter tuning with [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
- Track training time for each model
- Log all metrics and parameters to MLflow
- Display detailed results in Metaflow cards

### 5. Model Comparison (`join`)
- Compare performance of all models
- Analyze accuracy and training time
- Automatically select best model based on accuracy
- Visual comparison chart with color-coded training time
- Log best model to MLflow with signature and input example

## Key Features

- **Automatic Hyperparameter Optimization**: Uses hyperopt-sklearn for model tuning
- **Parallel Model Training**: Improves efficiency by training multiple models concurrently
- **Performance Tracking**: Measures both accuracy and training time
- **Interactive Charts**: Intuitive visual insights via Metaflow cards
- **Reusable Pipeline**: Data and models are version-controlled clearly

## Technologies Used

### Metaflow
- Workflow orchestration
- Parallel execution support
- Resource management and scalability
- Interactive chart support (cards)

### MLflow
- Experiment tracking
- Model management
- Data versioning
- Metric logging and visualization

### Hyperopt-sklearn
- Automated model selection and hyperparameter tuning
- Supports Bayesian (TPE), Random Search, Annealing, etc.
- Multiple scikit-learn algorithms supported
- Seamless integration with training pipeline and experiment tracking

## Setup

### Environment Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start MLflow server (in a separate terminal)
```bash
mlflow server --host 127.0.0.1 --port 5000
```

## Run Pipeline
### Basic run
```bash
python main.py run
```

### Enable Hyperparameter Optimization
```bash
python main.py run --use_hyperopt true
```
Note: Running with all models and tuning may take ~20 minutes.

### Use Custom Data Directory
```bash
python main.py run --data-dir /path/to/data
```

## Viewing results

### Giao Diện MLflow
```bash
mlflow ui
```
Access at: http://localhost:5000

### Metaflow Cards UI
In project directory, run in new terminal:
```bash
python main.py card server
```
Access at: http://localhost:8324

## Docker Compose Execution

### Build and Start Full System

In the project directory, run:

```bash
docker-compose up --build
```

- Service `mlflow-server`: MLflow Tracking Server at [http://localhost:5000](http://localhost:5000)
- Service `train-pipeline`: Automatically trains and registers best model in MLflow Model Registry
- Service `model-serving`: Serves best model via REST API at [http://localhost:5050/invocations](http://localhost:5050/invocations)

### 2. Send Inference Request

Once containers are running, send a prediction request to the model API:

```bash
curl -d '{"dataframe_split": {
"columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
"data": [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]}}' \
-H 'Content-Type: application/json' -X POST localhost:5050/invocations
```

Example response:

```json
{"predictions": [3]}
```

### Stop All Services

Press `Ctrl+C` in the running terminal or run:

```bash
docker-compose down
```

### Notes
- Ensure Docker and Docker Compose are installed.
- First run may take time to build images and download data.
- To view logs for a specific service:

```bash
docker-compose logs <service-name>
```

---

## Notes
- Ensure MLflow server is running before executing the pipeline
- First-time runs will download the dataset
- Hyperparameter tuning may significantly increase run time

# Monitoring Stack

## Components

- **Prometheus**: Metrics collection and storage (http://localhost:9090)
- **Grafana**: Metrics visualization (http://localhost:3000)
- **AlertManager**: Alert handling (http://localhost:9093)
- **Node Exporter**: System metrics collection
- **cAdvisor**: Container metrics collection
- **FastAPI Instrumentator**: API metrics collection

## Metrics Available

### System Metrics
- CPU usage
- Memory usage
- Disk I/O
- Network I/O
- Container metrics

### API Metrics
- Request rate
- Latency
- Error rates
- Status codes

### ML Model Metrics
- Inference latency
- Prediction confidence scores
- Model error rates

## Accessing Dashboards

1. **Grafana** (http://localhost:3000):
   - Default credentials: admin/admin
   - Pre-configured dashboards:
     - ML Model Metrics
     - System Metrics
     - API Metrics

2. **Prometheus** (http://localhost:9090):
   - Query metrics directly
   - View targets and alerts

3. **AlertManager** (http://localhost:9093):
   - View and manage alerts
   - Configure notifications

## Load Testing

Use the provided load testing script to generate traffic and test the monitoring stack:

```bash
# Run basic load test
python load_test.py

# Run with custom parameters
python load_test.py --duration 600 --workers 20 --rps 10

# Generate some errors for testing alerts
python load_test.py --inject-errors
```

## Alerts

The following alerts are configured:

1. **High Error Rate**
   - Triggers when error rate exceeds 50%
   - 5-minute evaluation window

2. **Low Model Confidence**
   - Triggers when average confidence falls below 0.6
   - 5-minute evaluation window

3. **High Latency**
   - Triggers when average prediction time exceeds 1 second
   - 5-minute evaluation window

Alerts are logged to `/alerts/alerts.log` and can be configured to send to Slack or email.

