@echo off
REM run_tests_docker.bat
REM Easy batch script to run pytest tests in Docker environment
REM Replicates the testenv Python 3.12.0 environment

setlocal enabledelayedexpansion

REM Set colors (if supported)
set "GREEN=[92m"
set "YELLOW=[93m" 
set "RED=[91m"
set "NC=[0m"

REM Function to print status
:print_status
echo %GREEN%[INFO]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARN]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Help function
:show_help
echo Usage: %0 [OPTION]
echo.
echo Run pytest tests in Docker environment (replicates testenv Python 3.12.0)
echo.
echo Options:
echo   all         Run all 119 tests (default)
echo   unit        Run unit tests only
echo   integration Run integration tests only
echo   performance Run performance tests only
echo   build       Build Docker image only
echo   shell       Open interactive shell in container
echo   clean       Remove Docker containers and images
echo   help        Show this help message
echo.
echo Examples:
echo   %0                    # Run all tests
echo   %0 unit              # Run unit tests only
echo   %0 shell             # Debug with interactive shell
echo   %0 clean             # Clean up Docker resources
goto :eof

REM Clean up function
:cleanup
call :print_status "Cleaning up Docker resources..."
docker-compose -f docker-compose.testenv.yml down --rmi all --volumes --remove-orphans >nul 2>&1
docker system prune -f >nul 2>&1
call :print_status "Cleanup complete"
goto :eof

REM Check if Docker is running
:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker and try again."
    exit /b 1
)
goto :eof

REM Main execution
set ACTION=%~1
if "%ACTION%"=="" set ACTION=all

if /i "%ACTION%"=="help" goto show_help
if /i "%ACTION%"=="--help" goto show_help
if /i "%ACTION%"=="-h" goto show_help

if /i "%ACTION%"=="clean" (
    call :cleanup
    exit /b 0
)

if /i "%ACTION%"=="build" (
    call :print_status "Building Docker image..."
    call :check_docker
    if errorlevel 1 exit /b 1
    docker-compose -f docker-compose.testenv.yml build
    if errorlevel 1 (
        call :print_error "Build failed"
        exit /b 1
    )
    call :print_status "Build complete"
    exit /b 0
)

if /i "%ACTION%"=="shell" (
    call :print_status "Opening interactive shell in testenv container..."
    call :check_docker
    if errorlevel 1 exit /b 1
    docker-compose -f docker-compose.testenv.yml build
    docker run --rm -it -v "%cd%:/app/Data-Pipeline" data-pipeline-testenv /bin/bash
    exit /b 0
)

call :check_docker
if errorlevel 1 exit /b 1

if /i "%ACTION%"=="all" (
    call :print_status "Running all 119 tests in Docker environment..."
    docker-compose -f docker-compose.testenv.yml up --build testenv
    set EXIT_CODE=!errorlevel!
) else if /i "%ACTION%"=="unit" (
    call :print_status "Running unit tests in Docker environment..."
    docker-compose -f docker-compose.testenv.yml up --build testenv-unit
    set EXIT_CODE=!errorlevel!
) else if /i "%ACTION%"=="integration" (
    call :print_status "Running integration tests in Docker environment..."
    docker-compose -f docker-compose.testenv.yml up --build testenv-integration
    set EXIT_CODE=!errorlevel!
) else if /i "%ACTION%"=="performance" (
    call :print_status "Running performance tests in Docker environment..."
    docker-compose -f docker-compose.testenv.yml up --build testenv-performance
    set EXIT_CODE=!errorlevel!
) else (
    call :print_error "Unknown option: %ACTION%"
    echo.
    call :show_help
    exit /b 1
)

REM Check exit code
if !EXIT_CODE! equ 0 (
    call :print_status "Tests completed successfully!"
    call :print_status "Check tests/logs/ for detailed results and coverage reports"
) else (
    call :print_error "Tests failed with exit code !EXIT_CODE!"
    call :print_warning "Check tests/logs/ for error details"
    exit /b !EXIT_CODE!
)

endlocal