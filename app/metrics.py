from prometheus_client import Counter, Histogram, Gauge
from typing import Optional

class MetricsSingleton:
    _instance: Optional['MetricsSingleton'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize metrics only once
            cls._instance.request_count = Counter(
                'http_requests_total',
                'Total number of HTTP requests',
                ['method', 'endpoint', 'status']
            )
            
            cls._instance.latency = Histogram(
                'model_prediction_seconds',
                'Time spent processing prediction'
            )
            
            cls._instance.confidence_score = Gauge(
                'model_confidence_score',
                'Model prediction confidence score'
            )
        return cls._instance

# Create singleton instance
metrics = MetricsSingleton()

# Add GPU timing metric
GPU_PREDICTION_TIME = Histogram(
    'model_prediction_gpu_seconds',
    'Time spent on GPU for prediction'
)