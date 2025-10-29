"""
Monitoring script for data acquisition pipeline
Tracks: processing time, data volume, error rates, resource utilization
"""

import time
import psutil
import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server, push_to_gateway, CollectorRegistry
from datetime import datetime
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
PIPELINE_PROCESSING_TIME = Histogram(
    'pipeline_processing_seconds',
    'Time spent processing pipeline stages',
    ['stage']
)

DATA_VOLUME_PROCESSED = Counter(
    'data_volume_bytes_total',
    'Total volume of data processed',
    ['stage']
)

PIPELINE_ERRORS = Counter(
    'pipeline_errors_total',
    'Total number of pipeline errors',
    ['stage', 'error_type']
)

CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage'
)

DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'Current disk usage percentage'
)

PIPELINE_STATUS = Gauge(
    'pipeline_status',
    'Pipeline status (1=running, 0=stopped, -1=failed)',
    ['pipeline_name']
)


class PipelineMonitor:
    """Main monitoring class for the data acquisition pipeline"""
    
    def __init__(self, port=8000, pushgateway_url='localhost:9091'):
        self.port = port
        self.pushgateway_url = pushgateway_url
        self.start_time = datetime.now()
        
    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(self.port)
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def track_stage_time(self, stage_name):
        """Decorator to track processing time for pipeline stages"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    PIPELINE_PROCESSING_TIME.labels(stage=stage_name).observe(duration)
                    logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.record_error(stage_name, type(e).__name__)
                    logger.error(f"Stage '{stage_name}' failed after {duration:.2f}s: {e}")
                    raise
            return wrapper
        return decorator
    
    def record_data_volume(self, stage_name, bytes_processed):
        """Record volume of data processed and push to Pushgateway"""
        DATA_VOLUME_PROCESSED.labels(stage=stage_name).inc(bytes_processed)
        logger.info(f"Stage '{stage_name}': Processed {bytes_processed / (1024**2):.2f} MB")
        
        # Push to Pushgateway for persistence (uses default registry)
        try:
            from prometheus_client import REGISTRY
            push_to_gateway(
                self.pushgateway_url,
                job=f'pipeline_{stage_name}',
                registry=REGISTRY
            )
        except Exception as e:
            logger.warning(f"Failed to push metrics to Pushgateway: {e}")
    
    def record_error(self, stage_name, error_type):
        """Record pipeline errors"""
        PIPELINE_ERRORS.labels(stage=stage_name, error_type=error_type).inc()
        logger.error(f"Error in stage '{stage_name}': {error_type}")
    
    def update_system_metrics(self):
        """Update system resource utilization metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.percent)
            
            logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def set_pipeline_status(self, pipeline_name, status):
        """
        Set pipeline status
        status: 1 (running), 0 (stopped), -1 (failed)
        """
        PIPELINE_STATUS.labels(pipeline_name=pipeline_name).set(status)
        logger.info(f"Pipeline '{pipeline_name}' status set to {status}")
    
    def continuous_monitoring(self, interval=60):
        """Continuously monitor system metrics"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        try:
            while True:
                self.update_system_metrics()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")


# Global monitor instance
monitor = PipelineMonitor()


# Example usage in your pipeline scripts
if __name__ == "__main__":
    # Start metrics server
    monitor.start_metrics_server()
    
    # Example: Track a pipeline stage
    @monitor.track_stage_time("data_ingestion")
    def ingest_data():
        time.sleep(2)  # Simulate processing
        return "data"
    
    # Run example
    monitor.set_pipeline_status("data_acquisition", 1)
    data = ingest_data()
    monitor.record_data_volume("data_ingestion", 1024 * 1024 * 50)  # 50 MB
    monitor.set_pipeline_status("data_acquisition", 0)
    
    # Start continuous monitoring
    monitor.continuous_monitoring(interval=30)