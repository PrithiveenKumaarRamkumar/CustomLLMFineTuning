# Updated Grafana Dashboard - Data Pipeline Monitoring

## ✅ Dashboard Updated Successfully

The Grafana dashboard has been updated to display the metrics that are **actually being collected** by your pipeline.

## Dashboard Panels

### 1. **Pipeline Status** (Top Banner)
- Shows the current status of each pipeline
- **Colors:**
  - 🟢 **Green** = Running (value: 1)
  - 🔵 **Blue** = Stopped (value: 0)
  - 🔴 **Red** = Failed (value: -1)
- **Metrics:** `pipeline_status{pipeline_name}`

### 2. **System Resource Gauges** (Row 1)
Three gauges showing real-time system metrics:

- **CPU Usage** - Current CPU utilization percentage
  - Query: `system_cpu_usage_percent`
  - Thresholds: Green (<70%), Yellow (70-85%), Red (>85%)

- **Memory Usage** - Current RAM utilization percentage
  - Query: `system_memory_usage_percent`
  - Thresholds: Green (<70%), Yellow (70-85%), Red (>85%)

- **Disk Usage** - Current disk space utilization percentage
  - Query: `system_disk_usage_percent`
  - Thresholds: Green (<70%), Yellow (70-85%), Red (>85%)

### 3. **System Resources Over Time** (Graph)
- Line graph showing CPU, Memory, and Disk trends
- Updates every 5 seconds
- Useful for identifying resource spikes during pipeline execution

### 4. **Pipeline Errors** (Bar Chart)
- Shows error counts by pipeline stage in the last hour
- Query: `sum(increase(pipeline_errors_total[1h])) by (stage)`
- Will populate when your pipeline encounters errors

### 5. **Data Volume Processed** (Time Series)
- Shows data throughput by pipeline stage
- Query: `sum(increase(data_volume_bytes_total[5m])) by (stage)`
- Displays in bytes/MB
- Will populate when you run pipeline stages that process data

## Currently Visible Metrics

Right now you should see:

✅ **Pipeline Status**
- Shows "data_acquisition" as Stopped (blue)
- Updates when you run pipelines

✅ **System Metrics** (Live Data)
- CPU: ~22-24%
- Memory: ~92-94%
- Disk: ~90%

❌ **Processing Time** - Removed (requires histogram metrics that aren't persisting)

❌ **Errors** - Empty (no errors logged yet)

❌ **Data Volume** - Empty (no pipeline runs with data volume tracking)

## How to Populate All Metrics

### Option 1: Run Sample Metrics (Quick Test)
```powershell
python scripts/generate_sample_metrics.py
```

This generates sample data for all metric types.

### Option 2: Run Actual Pipeline with Monitoring

The monitoring has been integrated into `scripts/dataset_filter.py`. Run it:

```powershell
python scripts/dataset_filter.py --input data/raw --output data/staged/raw_organized
```

This will generate:
- Pipeline status updates
- Error metrics (if errors occur)
- Data volume metrics
- Processing time (visible in logs, but histogram metrics don't persist in current setup)

### Option 3: Integrate Monitoring into Other Scripts

Add to any pipeline script:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring" / "scripts"))
from monitoring import monitor

# Set status
monitor.set_pipeline_status("my_pipeline", 1)  # Running

# Track stage execution time
@monitor.track_stage_time("my_stage")
def my_function():
    # your code
    pass

# Record data processed
monitor.record_data_volume("my_stage", bytes_processed)

# Record errors
try:
    risky_operation()
except Exception as e:
    monitor.record_error("my_stage", type(e).__name__)

# Complete
monitor.set_pipeline_status("my_pipeline", 0)  # Stopped
```

## Access the Dashboard

1. Open: **http://localhost:3000**
2. Login: `admin` / `admin`
3. Go to: **Dashboards → Browse → Data Pipeline Monitoring**

## Why Some Metrics Are Empty

**Processing Time removed:** The original dashboard had a processing time panel that used `pipeline_processing_seconds` histogram metrics. These only exist while a script is running and don't persist after it exits, making the dashboard panel always empty. I removed it to show only the metrics that actually work in your current setup.

**Errors/Data Volume empty:** These will populate when you run pipeline stages that:
- Encounter errors → generates `pipeline_errors_total`
- Process data → generates `data_volume_bytes_total`

The system resource metrics (CPU, Memory, Disk) work immediately because the metrics server continuously monitors system stats every 15 seconds.

## Dashboard Features

- **Auto-refresh**: Updates every 5 seconds
- **Time range**: Shows last 6 hours by default
- **Live updates**: All panels refresh automatically
- **Color-coded**: Status and gauges use intuitive color schemes

## Next Steps

1. ✅ View the dashboard now - system metrics are already visible
2. 🔄 Run `python scripts/generate_sample_metrics.py` to see all panels with data
3. 🔧 Run your actual data pipeline to see real metrics
4. 📊 Customize dashboard panels in Grafana UI as needed

The dashboard is now properly configured to show metrics that are actively being collected!
