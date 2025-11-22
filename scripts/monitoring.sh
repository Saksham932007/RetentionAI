#!/bin/bash

# RetentionAI Monitoring Stack Management Script
# This script helps manage the monitoring infrastructure for RetentionAI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.monitoring.yml"
PROJECT_NAME="retentionai-monitoring"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Start monitoring stack
start_monitoring() {
    print_status "Starting RetentionAI monitoring stack..."
    
    if [ -f "$COMPOSE_FILE" ]; then
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
        
        print_success "Monitoring stack started successfully!"
        print_status "Services are available at:"
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3000 (admin/retentionai123)"
        echo "  - Alertmanager: http://localhost:9093"
        echo "  - Node Exporter: http://localhost:9100"
        echo "  - cAdvisor: http://localhost:8080"
        echo "  - Loki: http://localhost:3100"
    else
        print_error "Docker Compose file '$COMPOSE_FILE' not found"
        exit 1
    fi
}

# Stop monitoring stack
stop_monitoring() {
    print_status "Stopping RetentionAI monitoring stack..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
    print_success "Monitoring stack stopped"
}

# Restart monitoring stack
restart_monitoring() {
    print_status "Restarting RetentionAI monitoring stack..."
    stop_monitoring
    start_monitoring
}

# Show status of monitoring services
show_status() {
    print_status "Checking status of monitoring services..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    
    print_status "Service health checks:"
    
    # Check Prometheus
    if curl -s -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_success "Prometheus: Healthy"
    else
        print_warning "Prometheus: Not healthy or not accessible"
    fi
    
    # Check Grafana
    if curl -s -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Grafana: Healthy"
    else
        print_warning "Grafana: Not healthy or not accessible"
    fi
    
    # Check Alertmanager
    if curl -s -f http://localhost:9093/-/healthy > /dev/null 2>&1; then
        print_success "Alertmanager: Healthy"
    else
        print_warning "Alertmanager: Not healthy or not accessible"
    fi
}

# View logs for specific service
show_logs() {
    if [ -z "$1" ]; then
        print_status "Available services:"
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps --services
        echo ""
        echo "Usage: $0 logs <service_name>"
        return 1
    fi
    
    print_status "Showing logs for $1..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$1"
}

# Clean up monitoring stack (removes volumes)
cleanup_monitoring() {
    print_warning "This will remove all monitoring data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up RetentionAI monitoring stack..."
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down -v
        print_success "Monitoring stack cleaned up"
    else
        print_status "Cleanup cancelled"
    fi
}

# Update monitoring stack
update_monitoring() {
    print_status "Updating RetentionAI monitoring stack..."
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" pull
    restart_monitoring
    print_success "Monitoring stack updated"
}

# Backup monitoring data
backup_monitoring() {
    BACKUP_DIR="monitoring_backup_$(date +%Y%m%d_%H%M%S)"
    print_status "Creating backup in $BACKUP_DIR..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup Prometheus data
    docker run --rm \
        -v "${PROJECT_NAME}_prometheus_data:/data" \
        -v "$(pwd)/$BACKUP_DIR:/backup" \
        busybox tar czf /backup/prometheus_data.tar.gz -C /data .
    
    # Backup Grafana data
    docker run --rm \
        -v "${PROJECT_NAME}_grafana_data:/data" \
        -v "$(pwd)/$BACKUP_DIR:/backup" \
        busybox tar czf /backup/grafana_data.tar.gz -C /data .
    
    # Backup Alertmanager data
    docker run --rm \
        -v "${PROJECT_NAME}_alertmanager_data:/data" \
        -v "$(pwd)/$BACKUP_DIR:/backup" \
        busybox tar czf /backup/alertmanager_data.tar.gz -C /data .
    
    print_success "Backup completed in $BACKUP_DIR/"
}

# Main script logic
main() {
    case "${1:-}" in
        start)
            check_dependencies
            start_monitoring
            ;;
        stop)
            stop_monitoring
            ;;
        restart)
            check_dependencies
            restart_monitoring
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        cleanup)
            cleanup_monitoring
            ;;
        update)
            check_dependencies
            update_monitoring
            ;;
        backup)
            backup_monitoring
            ;;
        *)
            echo "RetentionAI Monitoring Stack Management"
            echo ""
            echo "Usage: $0 {start|stop|restart|status|logs|cleanup|update|backup}"
            echo ""
            echo "Commands:"
            echo "  start     Start the monitoring stack"
            echo "  stop      Stop the monitoring stack"
            echo "  restart   Restart the monitoring stack"
            echo "  status    Show status of monitoring services"
            echo "  logs      Show logs for a specific service"
            echo "  cleanup   Remove monitoring stack and all data"
            echo "  update    Update monitoring stack images"
            echo "  backup    Backup monitoring data"
            echo ""
            echo "Examples:"
            echo "  $0 start"
            echo "  $0 logs prometheus"
            echo "  $0 status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"