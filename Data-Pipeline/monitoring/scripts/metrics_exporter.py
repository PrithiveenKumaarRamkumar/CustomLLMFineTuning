"""
Metrics exporter utility for custom metrics
"""

import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import logging

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Export custom metrics to Prometheus Pushgateway"""
    
    def __init__(self, pushgateway_url='localhost:9091', job_name='pipeline_batch'):
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()
        
    def export_batch_metrics(self, stage_name, records_processed, duration):
        """Export batch job metrics"""
        try:
            # Create metrics
            records_gauge = Gauge(
                'batch_records_processed',
                'Number of records processed in batch',
                ['stage'],
                registry=self.registry
            )
            duration_gauge = Gauge(
                'batch_duration_seconds',
                'Duration of batch processing',
                ['stage'],
                registry=self.registry
            )
            
            # Set values
            records_gauge.labels(stage=stage_name).set(records_processed)
            duration_gauge.labels(stage=stage_name).set(duration)
            
            # Push to gateway
            push_to_gateway(
                self.pushgateway_url,
                job=self.job_name,
                registry=self.registry
            )
            
            logger.info(f"Exported metrics for stage '{stage_name}'")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


if __name__ == "__main__":
    # Example usage
    exporter = MetricsExporter()
    exporter.export_batch_metrics("data_processing", 10000, 45.5)