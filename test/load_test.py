import requests
import random
import time
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5050/invocations"

def generate_wine_data():
    """Generate random wine data within reasonable ranges"""
    return {
        "dataframe_split": {
            "columns": [
                "fixed acidity", "volatile acidity", "citric acid", 
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
            ],
            "data": [[
                random.uniform(6.0, 9.0),      # fixed acidity
                random.uniform(0.1, 0.5),      # volatile acidity
                random.uniform(0.0, 0.5),      # citric acid
                random.uniform(1.0, 30.0),     # residual sugar
                random.uniform(0.02, 0.1),     # chlorides
                random.uniform(20.0, 70.0),    # free sulfur dioxide
                random.uniform(100.0, 200.0),  # total sulfur dioxide
                random.uniform(0.99, 1.01),    # density
                random.uniform(2.8, 3.8),      # pH
                random.uniform(0.3, 0.8),      # sulphates
                random.uniform(8.0, 14.0)      # alcohol
            ]]
        }
    }

def make_request(inject_error=False):
    """Make a request to the prediction endpoint"""
    try:
        data = generate_wine_data()
        headers = {"Content-Type": "application/json"}
        
        # Randomly inject errors 10% of the time if inject_error is True
        if inject_error and random.random() < 0.1:
            data["dataframe_split"]["data"][0][0] = "invalid"  # This will cause an error
            
        response = requests.post(BASE_URL, json=data, headers=headers)
        if response.status_code == 200:
            logger.info(f"Successful prediction: {response.json()}")
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")

def run_load_test(duration=300, max_workers=10, requests_per_second=5, inject_errors=False):
    """
    Run a load test for a specified duration
    
    Args:
        duration: Test duration in seconds (default: 300)
        max_workers: Maximum number of concurrent requests (default: 10)
        requests_per_second: Number of requests to make per second (default: 5)
        inject_errors: Whether to randomly inject errors (default: False)
    """
    logger.info(f"Starting load test for {duration} seconds...")
    end_time = time.time() + duration
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while time.time() < end_time:
            futures = []
            for _ in range(requests_per_second):
                futures.append(executor.submit(make_request, inject_errors))
                
            concurrent.futures.wait(futures)
            time.sleep(1)  # Wait 1 second before next batch
            
    logger.info("Load test completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load testing script for wine quality prediction API')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--workers', type=int, default=10, help='Maximum number of concurrent workers')
    parser.add_argument('--rps', type=int, default=5, help='Requests per second')
    parser.add_argument('--inject-errors', action='store_true', help='Randomly inject errors')
    
    args = parser.parse_args()
    
    run_load_test(
        duration=args.duration,
        max_workers=args.workers,
        requests_per_second=args.rps,
        inject_errors=args.inject_errors
    )
