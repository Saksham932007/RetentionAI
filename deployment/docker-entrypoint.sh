#!/bin/bash
# RetentionAI Docker Entrypoint Script
# Handles initialization, environment setup, and graceful startup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_info() {
    log "${BLUE}INFO${NC}: $1"
}

log_warn() {
    log "${YELLOW}WARN${NC}: $1"
}

log_error() {
    log "${RED}ERROR${NC}: $1"
}

log_success() {
    log "${GREEN}SUCCESS${NC}: $1"
}

# Environment setup
setup_environment() {
    log_info "Setting up RetentionAI environment..."
    
    # Set default values
    export ENVIRONMENT=${ENVIRONMENT:-development}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export WORKERS=${WORKERS:-1}
    export DATABASE_URL=${DATABASE_URL:-sqlite:///app/data/retentionai.db}
    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
    
    log_info "Environment: $ENVIRONMENT"
    log_info "Log Level: $LOG_LEVEL"
    log_info "Database: $DATABASE_URL"
    log_info "MLflow: $MLFLOW_TRACKING_URI"
}

# Database initialization
init_database() {
    log_info "Initializing database..."
    
    # Create data directory if it doesn't exist
    mkdir -p /app/data
    
    # Initialize database schema (if needed)
    if [ ! -f "/app/data/retentionai.db" ]; then
        log_info "Creating new database..."
        python -c "
from src.database import get_database_manager
try:
    db = get_database_manager()
    db.execute_query('CREATE TABLE IF NOT EXISTS health_check (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"
    else
        log_info "Database already exists"
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Check if required files exist
    required_files=(
        "/app/src/app.py"
        "/app/config.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check Python imports
    python -c "import streamlit; import pandas; import numpy; import plotly" || {
        log_error "Required Python packages not available"
        exit 1
    }
    
    log_success "Health check passed"
}

# Wait for dependencies
wait_for_deps() {
    if [ -n "${MLFLOW_TRACKING_URI:-}" ] && [[ "$MLFLOW_TRACKING_URI" =~ http.* ]]; then
        log_info "Waiting for MLflow server..."
        
        # Extract host and port from URI
        MLFLOW_HOST=$(echo $MLFLOW_TRACKING_URI | sed 's|http://||' | cut -d':' -f1)
        MLFLOW_PORT=$(echo $MLFLOW_TRACKING_URI | sed 's|http://||' | cut -d':' -f2)
        
        timeout 60 bash -c "until nc -z $MLFLOW_HOST $MLFLOW_PORT; do sleep 2; done" || {
            log_warn "MLflow server not available, continuing anyway..."
        }
        
        if nc -z $MLFLOW_HOST $MLFLOW_PORT; then
            log_success "MLflow server is available"
        fi
    fi
}

# Signal handlers for graceful shutdown
shutdown() {
    log_info "Received shutdown signal, gracefully stopping..."
    if [ -n "${STREAMLIT_PID:-}" ]; then
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
        wait "$STREAMLIT_PID" 2>/dev/null || true
    fi
    log_success "Shutdown complete"
    exit 0
}

trap 'shutdown' SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting RetentionAI application..."
    
    setup_environment
    health_check
    wait_for_deps
    init_database
    
    # Change to app directory
    cd /app
    
    # Start the application based on the service type
    case "${SERVICE_TYPE:-app}" in
        "app")
            log_info "Starting Streamlit application..."
            streamlit run src/app.py \
                --server.port=8501 \
                --server.address=0.0.0.0 \
                --server.headless=true \
                --server.enableCORS=false \
                --server.enableXsrfProtection=true \
                --logger.level=$LOG_LEVEL &
            STREAMLIT_PID=$!
            wait $STREAMLIT_PID
            ;;
        "training")
            log_info "Starting training pipeline..."
            python src/train.py
            ;;
        "api")
            log_info "Starting API server..."
            # This will be implemented when FastAPI is added
            uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers $WORKERS
            ;;
        *)
            log_error "Unknown service type: ${SERVICE_TYPE}"
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi