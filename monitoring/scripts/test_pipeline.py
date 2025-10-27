"""
Test pipeline to generate monitoring metrics
"""
import time
import random
from monitoring import monitor

# Start metrics server
monitor.start_metrics_server()

@monitor.track_stage_time("data_extraction")
def extract_data():
    """Simulate data extraction"""
    print("Extracting data...")
    time.sleep(random.uniform(1, 3))  # Simulate work
    data_size = random.randint(10, 100) * 1024 * 1024  # 10-100 MB
    monitor.record_data_volume("data_extraction", data_size)
    return f"extracted_{data_size}_bytes"

@monitor.track_stage_time("data_validation")
def validate_data(data):
    """Simulate data validation"""
    print("Validating data...")
    time.sleep(random.uniform(0.5, 2))
    
    # Simulate occasional errors
    if random.random() < 0.2:  # 20% chance of error
        monitor.record_error("data_validation", "ValidationError")
        raise ValueError("Validation failed")
    
    return data

@monitor.track_stage_time("data_transformation")
def transform_data(data):
    """Simulate data transformation"""
    print("Transforming data...")
    time.sleep(random.uniform(1, 4))
    data_size = random.randint(50, 150) * 1024 * 1024  # 50-150 MB
    monitor.record_data_volume("data_transformation", data_size)
    return f"transformed_{data}"

@monitor.track_stage_time("data_loading")
def load_data(data):
    """Simulate data loading"""
    print("Loading data...")
    time.sleep(random.uniform(0.5, 2))
    return "success"

def run_pipeline():
    """Run the complete pipeline"""
    pipeline_name = "test_data_pipeline"
    
    print(f"\n{'='*50}")
    print(f"Starting {pipeline_name}")
    print(f"{'='*50}\n")
    
    monitor.set_pipeline_status(pipeline_name, 1)  # Running
    
    try:
        data = extract_data()
        validated = validate_data(data)
        transformed = transform_data(validated)
        result = load_data(transformed)
        
        monitor.set_pipeline_status(pipeline_name, 0)  # Completed
        print(f"\n✓ Pipeline completed successfully: {result}\n")
        
    except Exception as e:
        monitor.set_pipeline_status(pipeline_name, -1)  # Failed
        print(f"\n✗ Pipeline failed: {e}\n")

if __name__ == "__main__":
    print("Starting test pipeline with monitoring...")
    print("Metrics available at: http://localhost:8000/metrics")
    print("View in Prometheus: http://localhost:9090")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Run pipeline multiple times to generate metrics
        for i in range(20):
            print(f"Pipeline run #{i+1}")
            try:
                run_pipeline()
            except Exception:
                pass  # Continue even if pipeline fails
            
            time.sleep(5)  # Wait between runs
            
    except KeyboardInterrupt:
        print("\n\nStopping test pipeline...")