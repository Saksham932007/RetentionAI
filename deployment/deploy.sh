#!/bin/bash
# RetentionAI Deployment Script
# Manages Docker-based deployments across different environments

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
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

# Default values
ENVIRONMENT="development"
COMMAND=""
SERVICES=""
BUILD=false
DOWN=false
LOGS=false
REBUILD=false

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND [SERVICES...]

Commands:
    up          Start services
    down        Stop and remove services
    restart     Restart services
    logs        View service logs
    build       Build service images
    rebuild     Rebuild and restart services
    status      Show service status
    clean       Clean up unused images and volumes
    shell       Open shell in service container

Options:
    -e, --env ENV       Environment (development|staging|production) [default: development]
    -b, --build         Build images before starting
    -d, --down          Stop services after command
    -f, --follow        Follow logs (for logs command)
    -h, --help          Show this help

Examples:
    $0 up                           # Start development environment
    $0 -e production up             # Start production environment
    $0 up retentionai-app mlflow    # Start specific services
    $0 rebuild                      # Rebuild and restart all services
    $0 logs -f retentionai-app      # Follow app logs
    $0 shell retentionai-app        # Open shell in app container

Profiles:
    development: app + mlflow
    staging:     app + mlflow + monitoring
    production:  app + mlflow + monitoring + nginx + api
    training:    trainer service for ML training jobs
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -d|--down)
            DOWN=true
            shift
            ;;
        -f|--follow)
            LOGS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        up|down|restart|logs|build|rebuild|status|clean|shell)
            COMMAND="$1"
            shift
            SERVICES="$*"
            break
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    log_error "Command required"
    usage
    exit 1
fi

# Environment validation
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production|training)
            log_info "Environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production, training"
            exit 1
            ;;
    esac
}

# Setup environment
setup_environment() {
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p data/{raw,processed} models logs
    
    # Generate SSL certificates if they don't exist
    if [[ ! -f "deployment/ssl/cert.pem" ]]; then
        log_info "Generating SSL certificates..."
        cd deployment/ssl
        bash generate-certs.sh
        cd "$PROJECT_ROOT"
    fi
    
    # Create environment file
    ENV_FILE=".env.${ENVIRONMENT}"
    if [[ ! -f "$ENV_FILE" ]]; then
        log_info "Creating environment file: $ENV_FILE"
        cat > "$ENV_FILE" << EOF
# RetentionAI ${ENVIRONMENT} Environment
ENVIRONMENT=${ENVIRONMENT}
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///app/data/retentionai.db
MLFLOW_TRACKING_URI=http://mlflow:5000
MODELS_DIR=/app/models
WORKERS=2
EOF
    fi
}

# Docker Compose command builder
get_compose_cmd() {
    local cmd="docker compose -f docker-compose.yml"
    
    # Add environment-specific overrides
    case "$ENVIRONMENT" in
        production)
            cmd="$cmd --profile production --profile monitoring --profile api"
            ;;
        staging)
            cmd="$cmd --profile monitoring"
            ;;
        training)
            cmd="$cmd --profile training"
            ;;
        development)
            # Default services only
            ;;
    esac
    
    # Add environment file
    if [[ -f ".env.${ENVIRONMENT}" ]]; then
        cmd="$cmd --env-file .env.${ENVIRONMENT}"
    fi
    
    echo "$cmd"
}

# Execute commands
execute_command() {
    local compose_cmd
    compose_cmd="$(get_compose_cmd)"
    
    case "$COMMAND" in
        up)
            if [[ "$BUILD" == "true" ]]; then
                log_info "Building services..."
                $compose_cmd build $SERVICES
            fi
            
            log_info "Starting services..."
            $compose_cmd up -d $SERVICES
            
            log_success "Services started successfully"
            $compose_cmd ps
            ;;
        down)
            log_info "Stopping services..."
            $compose_cmd down $SERVICES
            log_success "Services stopped"
            ;;
        restart)
            log_info "Restarting services..."
            $compose_cmd restart $SERVICES
            log_success "Services restarted"
            ;;
        logs)
            if [[ "$LOGS" == "true" ]]; then
                $compose_cmd logs -f $SERVICES
            else
                $compose_cmd logs --tail=100 $SERVICES
            fi
            ;;
        build)
            log_info "Building services..."
            $compose_cmd build $SERVICES
            log_success "Build completed"
            ;;
        rebuild)
            log_info "Rebuilding and restarting services..."
            $compose_cmd down $SERVICES
            $compose_cmd build --no-cache $SERVICES
            $compose_cmd up -d $SERVICES
            log_success "Services rebuilt and started"
            $compose_cmd ps
            ;;
        status)
            $compose_cmd ps
            echo ""
            $compose_cmd top
            ;;
        clean)
            log_info "Cleaning up unused resources..."
            docker system prune -f
            docker volume prune -f
            log_success "Cleanup completed"
            ;;
        shell)
            if [[ -z "$SERVICES" ]]; then
                log_error "Service name required for shell command"
                exit 1
            fi
            
            log_info "Opening shell in $SERVICES..."
            $compose_cmd exec "$SERVICES" /bin/bash
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    if [[ "$DOWN" == "true" ]] && [[ "$COMMAND" != "down" ]]; then
        log_info "Stopping services..."
        $compose_cmd down
    fi
}

# Health check
health_check() {
    if [[ "$COMMAND" == "up" ]] || [[ "$COMMAND" == "restart" ]] || [[ "$COMMAND" == "rebuild" ]]; then
        log_info "Performing health check..."
        sleep 5
        
        # Check if services are healthy
        local compose_cmd
        compose_cmd="$(get_compose_cmd)"
        
        if $compose_cmd ps --filter "health=unhealthy" | grep -q "unhealthy"; then
            log_warn "Some services are unhealthy:"
            $compose_cmd ps --filter "health=unhealthy"
        else
            log_success "All services are healthy"
        fi
    fi
}

# Cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed"
    fi
}

trap cleanup EXIT

# Main execution
main() {
    log_info "RetentionAI Deployment Manager"
    
    validate_environment
    setup_environment
    execute_command
    health_check
    
    log_success "Deployment completed successfully"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi