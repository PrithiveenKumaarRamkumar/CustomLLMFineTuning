#!/bin/bash

# Data Pipeline Quality Assurance Runner
# Comprehensive test execution with quality gates and reporting

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$PROJECT_ROOT/quality_reports"
CONFIG_FILE="$PROJECT_ROOT/configs/quality_gates_config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        log_error "pytest is required but not installed. Run: pip install pytest"
        exit 1
    fi
    
    # Check coverage
    if ! python3 -c "import coverage" &> /dev/null; then
        log_error "coverage is required but not installed. Run: pip install coverage"
        exit 1
    fi
    
    # Check other required packages
    local required_packages=("pandas" "matplotlib" "seaborn" "yaml")
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            log_warning "$package not found, some features may be limited"
        fi
    done
    
    log_success "Dependencies check completed"
}

# Setup directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$REPORTS_DIR/coverage"
    mkdir -p "$REPORTS_DIR/artifacts"
    
    # Create logs directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/logs"
    
    log_success "Directories setup completed"
}

# Run comprehensive test suite
run_test_suite() {
    local test_type="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    log_info "Running $test_type tests..."
    
    cd "$PROJECT_ROOT"
    
    case "$test_type" in
        "unit")
            pytest tests/unit/ \
                -v \
                --tb=short \
                --cov=scripts \
                --cov-report=html:quality_reports/coverage/unit_coverage \
                --cov-report=json:quality_reports/unit_coverage.json \
                --json-report \
                --json-report-file=quality_reports/unit_test_results.json \
                -m unit \
                --maxfail=10 \
                --durations=10
            ;;
        "integration")
            pytest tests/integration/ \
                -v \
                --tb=short \
                --json-report \
                --json-report-file=quality_reports/integration_test_results.json \
                -m integration \
                --maxfail=5 \
                --durations=10
            ;;
        "performance")
            pytest tests/performance/ \
                -v \
                --tb=short \
                --json-report \
                --json-report-file=quality_reports/performance_test_results.json \
                -m performance \
                --maxfail=3 \
                --durations=10
            ;;
        "bias")
            pytest tests/bias/ \
                -v \
                --tb=short \
                --json-report \
                --json-report-file=quality_reports/bias_test_results.json \
                -m bias \
                --maxfail=5 \
                --durations=10
            ;;
        "anomaly")
            pytest tests/anomaly/ \
                -v \
                --tb=short \
                --json-report \
                --json-report-file=quality_reports/anomaly_test_results.json \
                -m anomaly \
                --maxfail=5 \
                --durations=10
            ;;
        "all"|"comprehensive")
            pytest tests/ \
                -v \
                --tb=short \
                --cov=scripts \
                --cov-report=html:quality_reports/coverage/comprehensive_coverage \
                --cov-report=json:quality_reports/comprehensive_coverage.json \
                --cov-report=xml:quality_reports/coverage.xml \
                --json-report \
                --json-report-file=quality_reports/comprehensive_test_results.json \
                --maxfail=20 \
                --durations=20
            ;;
        *)
            log_error "Unknown test type: $test_type"
            exit 1
            ;;
    esac
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "$test_type tests completed successfully"
    else
        log_warning "$test_type tests completed with issues (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Run quality gates analysis
run_quality_gates() {
    log_info "Running quality gates analysis..."
    
    cd "$PROJECT_ROOT"
    
    # Run the quality gates script
    python3 scripts/quality_gates.py
    local exit_code=$?
    
    case $exit_code in
        0)
            log_success "All quality gates passed"
            ;;
        1)
            log_error "Critical quality gates failed"
            ;;
        2)
            log_warning "Some quality gates failed (warnings)"
            ;;
        *)
            log_error "Quality gates analysis failed with exit code: $exit_code"
            ;;
    esac
    
    return $exit_code
}

# Run anomaly detection
run_anomaly_detection() {
    log_info "Running anomaly detection analysis..."
    
    cd "$PROJECT_ROOT"
    
    if [ -f "scripts/anomaly_detection.py" ]; then
        python3 -c "
from scripts.anomaly_detection import PipelineAnomalyDetector
import json

detector = PipelineAnomalyDetector()
results = detector.detect_pipeline_anomalies('data/processed/')
print(json.dumps(results, indent=2))
" > quality_reports/anomaly_detection_results.json
        
        log_success "Anomaly detection analysis completed"
    else
        log_warning "Anomaly detection script not found, skipping..."
    fi
}

# Run bias analysis
run_bias_analysis() {
    log_info "Running bias analysis..."
    
    cd "$PROJECT_ROOT"
    
    if [ -f "scripts/bias_analysis.py" ]; then
        python3 -c "
from scripts.bias_analysis import ComprehensiveBiasAnalyzer
import json
import pandas as pd

# Create sample dataset for analysis
analyzer = ComprehensiveBiasAnalyzer()

# Mock some sample data for analysis
sample_data = pd.DataFrame({
    'language': ['python', 'javascript', 'java', 'python', 'go'],
    'stars': [100, 50, 200, 80, 120],
    'size': [1000, 500, 2000, 800, 1200],
    'has_pii': [False, True, False, False, True]
})

results = analyzer.analyze_comprehensive_bias(sample_data)
print(json.dumps(results, indent=2, default=str))
" > quality_reports/bias_analysis_results.json
        
        log_success "Bias analysis completed"
    else
        log_warning "Bias analysis script not found, skipping..."
    fi
}

# Generate summary report
generate_summary_report() {
    log_info "Generating summary report..."
    
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local report_file="quality_reports/qa_summary_$(date +"%Y%m%d_%H%M%S").md"
    
    cat > "$report_file" << EOF
# Data Pipeline Quality Assurance Summary

**Generated:** $timestamp

## Test Execution Results

EOF
    
    # Add test results if available
    if [ -f "quality_reports/comprehensive_test_results.json" ]; then
        echo "### Comprehensive Test Results" >> "$report_file"
        python3 -c "
import json
with open('quality_reports/comprehensive_test_results.json', 'r') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f\"- Total Tests: {summary.get('total', 0)}\")
    print(f\"- Passed: {summary.get('passed', 0)}\")
    print(f\"- Failed: {summary.get('failed', 0)}\")
    print(f\"- Skipped: {summary.get('skipped', 0)}\")
    if summary.get('total', 0) > 0:
        success_rate = (summary.get('passed', 0) / summary.get('total', 1)) * 100
        print(f\"- Success Rate: {success_rate:.1f}%\")
" >> "$report_file" 2>/dev/null || echo "- Test results not available" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add coverage information
    if [ -f "quality_reports/comprehensive_coverage.json" ]; then
        echo "### Code Coverage" >> "$report_file"
        python3 -c "
import json
with open('quality_reports/comprehensive_coverage.json', 'r') as f:
    data = json.load(f)
    totals = data.get('totals', {})
    print(f\"- Coverage: {totals.get('percent_covered', 0):.1f}%\")
    print(f\"- Lines Covered: {totals.get('covered_lines', 0)}\")
    print(f\"- Total Lines: {totals.get('num_statements', 0)}\")
" >> "$report_file" 2>/dev/null || echo "- Coverage information not available" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add quality gates status
    echo "### Quality Gates Status" >> "$report_file"
    echo "- Quality gates analysis completed" >> "$report_file"
    echo "- See detailed quality report for full results" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add recommendations
    echo "### Recommendations" >> "$report_file"
    echo "- Review detailed test reports in the quality_reports directory" >> "$report_file"
    echo "- Check code coverage reports for areas needing more tests" >> "$report_file"
    echo "- Address any quality gate failures highlighted in the reports" >> "$report_file"
    echo "- Monitor bias and anomaly detection results regularly" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add file locations
    echo "### Generated Reports" >> "$report_file"
    echo "- Summary Report: $report_file" >> "$report_file"
    echo "- Coverage Report: quality_reports/coverage/" >> "$report_file"
    echo "- Test Results: quality_reports/*_test_results.json" >> "$report_file"
    echo "- Quality Report: quality_reports/quality_report_*.html" >> "$report_file"
    
    log_success "Summary report generated: $report_file"
}

# Cleanup old reports
cleanup_old_reports() {
    log_info "Cleaning up old reports..."
    
    # Keep reports from last 30 days
    find "$REPORTS_DIR" -name "*.json" -mtime +30 -delete 2>/dev/null || true
    find "$REPORTS_DIR" -name "*.html" -mtime +30 -delete 2>/dev/null || true
    find "$REPORTS_DIR" -name "*.md" -mtime +30 -delete 2>/dev/null || true
    
    log_success "Old reports cleanup completed"
}

# Main execution function
main() {
    local test_type="${1:-comprehensive}"
    local skip_quality_gates="${2:-false}"
    
    echo "=============================================="
    echo "🔍 Data Pipeline Quality Assurance Runner"
    echo "=============================================="
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Setup directories
    setup_directories
    
    # Cleanup old reports
    cleanup_old_reports
    
    # Record start time
    local start_time=$(date +%s)
    
    # Initialize exit code tracking
    local overall_exit_code=0
    
    # Run test suites
    case "$test_type" in
        "unit"|"integration"|"performance"|"bias"|"anomaly")
            run_test_suite "$test_type" || overall_exit_code=$?
            ;;
        "all"|"comprehensive"|*)
            # Run all test suites
            run_test_suite "unit" || overall_exit_code=$?
            run_test_suite "integration" || overall_exit_code=$?
            run_test_suite "performance" || overall_exit_code=$?
            run_test_suite "bias" || overall_exit_code=$?
            run_test_suite "anomaly" || overall_exit_code=$?
            
            # Run comprehensive suite for overall coverage
            run_test_suite "comprehensive" || overall_exit_code=$?
            ;;
    esac
    
    # Run anomaly detection
    run_anomaly_detection
    
    # Run bias analysis
    run_bias_analysis
    
    # Run quality gates (unless skipped)
    if [ "$skip_quality_gates" != "true" ]; then
        run_quality_gates || overall_exit_code=$?
    fi
    
    # Generate summary report
    generate_summary_report
    
    # Calculate execution time
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    
    echo ""
    echo "=============================================="
    echo "🏁 Quality Assurance Execution Complete"
    echo "=============================================="
    echo "Execution Time: ${execution_time}s"
    echo "Reports Directory: $REPORTS_DIR"
    
    if [ $overall_exit_code -eq 0 ]; then
        log_success "All quality assurance checks passed!"
    elif [ $overall_exit_code -eq 2 ]; then
        log_warning "Quality assurance completed with warnings"
    else
        log_error "Quality assurance failed with critical issues"
    fi
    
    echo ""
    echo "📄 View detailed reports:"
    echo "  - HTML Reports: quality_reports/*.html"
    echo "  - Coverage Reports: quality_reports/coverage/"
    echo "  - Test Results: quality_reports/*_test_results.json"
    echo ""
    
    exit $overall_exit_code
}

# Help function
show_help() {
    cat << EOF
Data Pipeline Quality Assurance Runner

Usage: $0 [TEST_TYPE] [OPTIONS]

TEST_TYPE:
    unit            Run unit tests only
    integration     Run integration tests only
    performance     Run performance tests only
    bias            Run bias analysis tests only
    anomaly         Run anomaly detection tests only
    comprehensive   Run all tests (default)
    all             Same as comprehensive

OPTIONS:
    --skip-gates    Skip quality gates evaluation
    --help         Show this help message

Examples:
    $0                          # Run comprehensive test suite
    $0 unit                     # Run unit tests only
    $0 comprehensive --skip-gates   # Run all tests but skip quality gates
    $0 --help                   # Show help

Environment Variables:
    QA_CONFIG_FILE             Path to quality gates configuration file
    QA_REPORTS_DIR             Directory for quality reports
    QA_SKIP_CLEANUP            Skip cleanup of old reports (set to 'true')

EOF
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Check for skip-gates option
skip_gates="false"
if [ "$2" = "--skip-gates" ] || [ "$1" = "--skip-gates" ]; then
    skip_gates="true"
fi

# Run main function
main "${1:-comprehensive}" "$skip_gates"