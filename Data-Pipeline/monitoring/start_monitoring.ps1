# Monitoring Control Script
# Usage: .\monitoring\start_monitoring.ps1

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Data Pipeline Monitoring Stack" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Start monitoring containers
Write-Host "`nStarting monitoring containers..." -ForegroundColor Yellow
Set-Location monitoring
docker-compose -f docker-compose.monitoring.yml up -d
Set-Location ..

Start-Sleep -Seconds 5

# Check container status
Write-Host "`nContainer Status:" -ForegroundColor Yellow
docker ps --filter "name=pipeline_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Start metrics server in new window
Write-Host "`nStarting metrics server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python monitoring\scripts\start_metrics_server.py"

Start-Sleep -Seconds 3

# Verify metrics server
try {
    $response = Invoke-WebRequest -Uri http://localhost:8000/metrics -UseBasicParsing -TimeoutSec 5
    Write-Host "✓ Metrics server is running on http://localhost:8000/metrics" -ForegroundColor Green
} catch {
    Write-Host "✗ Metrics server failed to start" -ForegroundColor Red
}

Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "Monitoring Stack Started Successfully!" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`nAccess Points:" -ForegroundColor Yellow
Write-Host "  Grafana:     http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor White
Write-Host "  Metrics:     http://localhost:8000/metrics" -ForegroundColor White
Write-Host "  Alertmanager: http://localhost:9093" -ForegroundColor White

Write-Host "`nNote: Metrics server is running in a separate window. Close that window to stop it." -ForegroundColor Cyan
Write-Host "To stop containers: docker-compose -f monitoring\docker-compose.monitoring.yml down`n" -ForegroundColor Cyan
