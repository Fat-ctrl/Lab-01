import os
import time
import logging
import logging.config
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
from prometheus_client import Histogram, Counter, Gauge, Summary, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

# --- Logging setup ---
logging.config.fileConfig('app/logging.conf')
logger = logging.getLogger('app')
syslog_logger = logging.getLogger('syslog')

# --- Singleton Metrics class ---
class Metrics:
    _instance: Optional['Metrics'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.prediction_time = Histogram(
                "model_prediction_seconds", "Time spent processing prediction",
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )
            cls._instance.confidence_score = Gauge(
                "model_confidence_score", "Confidence score of predictions"
            )
            cls._instance.prediction_errors = Counter(
                "model_prediction_errors_total", "Total count of prediction errors"
            )
            cls._instance.predictions_total = Counter(
                "model_predictions_total", "Total number of predictions", ["result"]
            )
            cls._instance.batch_size = Summary(
                "model_batch_size", "Summary of batch sizes"
            )
            cls._instance.model_response_time = Summary(
                "model_response_time_seconds", "Model response time in seconds"
            )
            cls._instance.request_count = Counter(
                'http_requests_total', 'Total number of HTTP requests',
                ['method', 'endpoint', 'status']
            )
            cls._instance.gpu_prediction_time = Histogram(
                'model_prediction_gpu_seconds', 'Time spent on GPU for prediction'
            )
        return cls._instance

    def log_request(self, request: Request, response: Response):
        self.request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

# --- App init ---
metrics = Metrics()
app = FastAPI(title="Wine Quality Prediction Service")
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up")
    syslog_logger.info("ML service starting up")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    syslog_logger.error(f"Application error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

# --- Middleware to auto-log HTTP requests ---
@app.middleware("http")
async def log_metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    metrics.log_request(request, response)
    return response

# --- Input schemas ---
class DataFrameSplit(BaseModel):
    columns: List[str]
    data: List[List[float]]

class PredictionRequest(BaseModel):
    dataframe_split: DataFrameSplit

# --- Load model from MLflow ---
def load_model():
    try:
        model_uri = os.getenv("MODEL_URI", "models:/sk-learn-best-model/latest")
        logger.info(f"Loading model from {model_uri}")
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

model = None
try:
    model = load_model()
except Exception as e:
    logger.error("Model loading failed.")
    raise e

# --- Prediction Endpoint ---
@app.post("/invocations")
async def predict(request: PredictionRequest):
    try:
        start_time = time.time()

        df = pd.DataFrame(
            request.dataframe_split.data, 
            columns=request.dataframe_split.columns
        )
        metrics.batch_size.observe(len(df))

        with metrics.prediction_time.time():
            predictions = model.predict(df)

        # Confidence score calculation
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)
                confidence = float(np.max(proba, axis=1).mean())
            elif hasattr(model, 'decision_function'):
                # For models like SVM that use decision_function instead
                decision_scores = model.decision_function(df)
                if decision_scores.ndim > 1:
                    # Multi-class case
                    confidence = float(np.max(softmax(decision_scores, axis=1), axis=1).mean())
                else:
                    # Binary case
                    confidence = float(np.mean(np.abs(decision_scores)))
            else:
                # Fallback for models without probability estimates
                confidence = float(np.mean([1.0 if pred == true_label else 0.0 
                                         for pred, true_label in zip(predictions, df['quality'])]))
            
            logger.info(f"Predicted confidence score: {confidence:.4f}")
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {str(e)}")
            confidence = 0.0

        metrics.confidence_score.set(confidence)
        metrics.predictions_total.labels(result="success").inc()

        elapsed = time.time() - start_time
        metrics.model_response_time.observe(elapsed)

        return {
            "predictions": predictions.tolist(),
            "confidence": confidence
        }

    except Exception as e:
        metrics.prediction_errors.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add softmax function for multi-class confidence calculation
def softmax(x, axis=None):
    """Compute softmax values for each set of scores in x."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# --- Health & metrics endpoints ---
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/raw-metrics")
def get_raw_metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
