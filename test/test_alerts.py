import requests
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertTester:
    def __init__(self, api_url="http://localhost:5050"):
        self.api_url = api_url
        self.endpoint = f"{api_url}/invocations"

    def _make_request(self, delay=0):
        """Make a single request with optional artificial delay"""
        data = {
            "dataframe_split": {
                "columns": ["feature1", "feature2"],
                "data": [[1.0, 2.0], [3.0, 4.0]]
            }
        }
        
        try:
            time.sleep(delay)  # Artificial delay
            response = requests.post(self.endpoint, json=data)
            return response.status_code
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return 500

    def test_high_latency(self, duration=300, delay=1.0):
        """Test high latency alert by adding artificial delay"""
        logger.info(f"Testing high latency alert with {delay}s delay for {duration}s")
        start_time = time.time()
        request_count = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            while time.time() - start_time < duration:
                executor.submit(self._make_request, delay)
                request_count += 1
                
                # Add progress updates every 5 seconds
                elapsed = time.time() - start_time
                if request_count % 10 == 0:
                    logger.info(f"Progress: {elapsed:.1f}s / {duration}s - Requests made: {request_count}")
                
                time.sleep(0.1)  # Prevent overwhelming the server
        
        logger.info(f"Completed {request_count} requests over {duration} seconds")

    def test_error_rate(self, duration=300, error_rate=0.6):
        """Test error rate alert by generating errors"""
        logger.info(f"Testing error rate alert with {error_rate*100}% errors for {duration}s")
        start_time = time.time()
        request_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            while time.time() - start_time < duration:
                # Simulate errors based on error_rate
                if time.time() % 1 < error_rate:
                    # Send malformed request to generate 400 error
                    requests.post(self.endpoint, json={})
                    error_count += 1
                else:
                    executor.submit(self._make_request)
                request_count += 1
                
                # Add progress updates every 5 seconds
                elapsed = time.time() - start_time
                if request_count % 10 == 0:
                    logger.info(f"Progress: {elapsed:.1f}s / {duration}s - Requests: {request_count} (Errors: {error_count})")
                
                time.sleep(0.1)
        
        logger.info(f"Completed {request_count} requests ({error_count} errors) over {duration} seconds")

def main():
    tester = AlertTester()
    
    try:
        # Test high latency alert with 600ms delay (above 500ms threshold)
        logger.info("Starting high latency test...")
        tester.test_high_latency(duration=30, delay=2)
        logger.info("High latency test completed")
        
        time.sleep(30)  # Reduced wait time
        
        # Test error rate alert with shorter duration
        logger.info("Starting error rate test...")
        tester.test_error_rate(duration=30, error_rate=0.6)  # Reduced from 120s to 30s
        logger.info("Error rate test completed")
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()