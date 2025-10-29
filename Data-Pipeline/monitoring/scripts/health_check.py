"""
Health check script for pipeline components
"""

import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Check health of pipeline services"""
    
    def __init__(self):
        self.services = {
            'metrics_server': 'http://localhost:8000/metrics',
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3000/api/health'
        }
    
    def check_service(self, name, url, timeout=5):
        """Check if a service is healthy"""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                logger.info(f"✓ {name} is healthy")
                return True
            else:
                logger.warning(f"✗ {name} returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ {name} is unreachable: {e}")
            return False
    
    def check_all(self):
        """Check all services"""
        logger.info(f"Health check started at {datetime.now()}")
        results = {}
        
        for name, url in self.services.items():
            results[name] = self.check_service(name, url)
        
        # Summary
        healthy = sum(results.values())
        total = len(results)
        logger.info(f"\nHealth Check Summary: {healthy}/{total} services healthy")
        
        return all(results.values())


if __name__ == "__main__":
    checker = HealthChecker()
    all_healthy = checker.check_all()
    exit(0 if all_healthy else 1)