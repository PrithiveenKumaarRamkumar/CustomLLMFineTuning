#!/bin/bash

# run_tests_docker.sh
# Easy script to run pytest tests in Docker environment
# Replicates the testenv Python 3.12.0 environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Run pytest tests in Docker environment (replicates testenv Python 3.12.0)"
    echo ""
    echo "Options:"
    echo "  all         Run all 119 tests (default)"
    echo "  unit        Run unit tests only"
    echo "  integration Run integration tests only" 
    echo "  performance Run performance tests only"
    echo "  build       Build Docker image only"
    echo "  shell       Open interactive shell in container"
    echo "  clean       Remove Docker containers and images"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run all tests"
    echo "  $0 unit              # Run unit tests only"
    echo "  $0 shell             # Debug with interactive shell"
    echo "  $0 clean             # Clean up Docker resources"
}

# Clean up function
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose -f docker-compose.testenv.yml down --rmi all --volumes --remove-orphans 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
    print_status "Cleanup complete"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Main execution
main() {
    local action=${1:-all}
    
    case $action in
        help|--help|-h)
            show_help
            exit 0
            ;;
        clean)
            cleanup
            exit 0
            ;;
        build)
            print_status "Building Docker image..."
            check_docker
            docker-compose -f docker-compose.testenv.yml build
            print_status "Build complete"
            exit 0
            ;;
        shell)
            print_status "Opening interactive shell in testenv container..."
            check_docker
            docker-compose -f docker-compose.testenv.yml build
            docker run --rm -it -v "$(pwd):/app/Data-Pipeline" \
                $(docker-compose -f docker-compose.testenv.yml config | grep 'image:' | awk '{print $2}' | head -1 || echo "data-pipeline-testenv") \
                /bin/bash
            exit 0
            ;;
        all)
            print_status "Running all 119 tests in Docker environment..."
            check_docker
            docker-compose -f docker-compose.testenv.yml up --build testenv
            ;;
        unit)
            print_status "Running unit tests in Docker environment..."
            check_docker
            docker-compose -f docker-compose.testenv.yml up --build testenv-unit
            ;;
        integration)
            print_status "Running integration tests in Docker environment..."
            check_docker
            docker-compose -f docker-compose.testenv.yml up --build testenv-integration
            ;;
        performance)
            print_status "Running performance tests in Docker environment..."
            check_docker
            docker-compose -f docker-compose.testenv.yml up --build testenv-performance
            ;;
        *)
            print_error "Unknown option: $action"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    # Check exit code
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        print_status "Tests completed successfully!"
        print_status "Check tests/logs/ for detailed results and coverage reports"
    else
        print_error "Tests failed with exit code $exit_code"
        print_warning "Check tests/logs/ for error details"
        exit $exit_code
    fi
}

# Execute main function
main "$@"