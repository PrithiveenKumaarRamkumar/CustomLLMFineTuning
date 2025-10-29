"""
Start monitoring metrics server for the data pipeline.
Run this in a separate terminal or as a background service.
"""

import sys
import time
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from monitoring import monitor

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Data Pipeline Metrics Server")
    print("=" * 70)
    print(f"Metrics endpoint: http://localhost:8000/metrics")
    print(f"Prometheus scraping interval: 15s")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    # Start metrics HTTP server
    monitor.start_metrics_server()
    
    # Set initial status
    monitor.set_pipeline_status("data_acquisition", 0)  # Stopped initially
    
    # Run continuous system monitoring
    try:
        monitor.continuous_monitoring(interval=15)
    except KeyboardInterrupt:
        print("\nMetrics server stopped.")
        sys.exit(0)
