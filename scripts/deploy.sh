#!/bin/bash

# RetentionAI Production Deployment Script
# This script sets up a complete production deployment with monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
APP_NAME="retentionai"
COMPOSE_PROJECT="retentionai-production"
NETWORK_NAME="retentionai-network"
DATA_DIR="/opt/retentionai/data"
LOG_DIR="/opt/retentionai/logs"
BACKUP_DIR="/opt/retentionai/backups"

# Print colored output
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BOLD}${BLUE}=== $1 ===${NC}"; }

# Check if running as root for production setup
check_privileges() {
    if [[ $EUID -ne 0 ]] && [[ "$1" == "production" ]]; then
        print_error "Production deployment requires root privileges"
        print_status "Run with: sudo $0 production"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose available"
    
    # Check available disk space (minimum 10GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space: $(( available_space / 1024 / 1024 ))GB available, 10GB recommended"
    else
        print_success "Disk space: $(( available_space / 1024 / 1024 ))GB available"
    fi
    
    # Check available memory (minimum 4GB)
    available_memory=$(free -m | awk 'NR==2{print $2}')
    required_memory=4096  # 4GB in MB
    
    if [ "$available_memory" -lt "$required_memory" ]; then
        print_warning "Low memory: ${available_memory}MB available, 4GB recommended"
    else
        print_success "Memory: ${available_memory}MB available"
    fi
}

# Setup production directories
setup_directories() {
    print_header "Setting Up Production Directories"
    
    directories=(
        "$DATA_DIR"
        "$DATA_DIR/raw"
        "$DATA_DIR/processed"
        "$DATA_DIR/models"
        "$LOG_DIR"
        "$LOG_DIR/application"
        "$LOG_DIR/monitoring"
        "$BACKUP_DIR"
        "$BACKUP_DIR/database"
        "$BACKUP_DIR/models"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_status "Directory exists: $dir"
        fi
    done
    
    # Set proper permissions
    chown -R $(whoami):$(whoami) /opt/retentionai
    chmod -R 755 /opt/retentionai
    print_success "Set directory permissions"
}

# Create production environment file
create_production_env() {
    print_header "Creating Production Environment Configuration"
    
    cat > .env.production << EOF
# RetentionAI Production Environment Configuration
ENVIRONMENT=production

# Application Configuration
APP_NAME=retentionai
APP_VERSION=1.0.0
DEBUG=false

# Database Configuration
DATABASE_URL=sqlite:///data/retentionai.db
DATABASE_POOL_SIZE=10
DATABASE_TIMEOUT=30

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=retentionai-production
MLFLOW_ARTIFACT_ROOT=/opt/retentionai/data/mlflow

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERTMANAGER_PORT=9093

# Health Check Configuration
HEALTH_CHECK_PORT=8001
METRICS_PORT=8000

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=/opt/retentionai/logs

# Security Configuration
SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Performance Configuration
WORKERS=4
MAX_CONNECTIONS=100
REQUEST_TIMEOUT=30

# Backup Configuration
BACKUP_DIR=/opt/retentionai/backups
BACKUP_RETENTION_DAYS=30
EOF
    
    print_success "Created production environment file"
}

# Create production Docker Compose
create_production_compose() {
    print_header "Creating Production Docker Compose Configuration"
    
    cat > docker-compose.production.yml << 'EOF'
version: '3.8'

services:
  # Main RetentionAI Application
  retentionai-app:
    image: retentionai:latest
    container_name: retentionai-app
    restart: unless-stopped
    ports:
      - "8501:8501"  # Streamlit
      - "8000:8000"  # Metrics
      - "8001:8001"  # Health checks
    volumes:
      - /opt/retentionai/data:/app/data
      - /opt/retentionai/logs/application:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    networks:
      - retentionai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    depends_on:
      - prometheus
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.retentionai.rule=Host(`localhost`)"
      - "traefik.http.services.retentionai.loadbalancer.server.port=8501"

  # MLflow Tracking Server
  mlflow:
    image: python:3.10-slim
    container_name: retentionai-mlflow
    restart: unless-stopped
    ports:
      - "5000:5000"
    volumes:
      - /opt/retentionai/data/mlflow:/mlflow
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    networks:
      - retentionai-network
    command: >
      bash -c "
        pip install mlflow==2.8.0 &&
        mlflow server
        --backend-store-uri sqlite:////mlflow/mlflow.db
        --default-artifact-root /mlflow/artifacts
        --host 0.0.0.0
        --port 5000
      "
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus - Metrics Collection
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: retentionai-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./config/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - retentionai-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Grafana - Visualization
  grafana:
    image: grafana/grafana:10.0.0
    container_name: retentionai-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-retentionai123}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SECURITY_DISABLE_GRAVATAR=true
      - GF_ANALYTICS_REPORTING_ENABLED=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana-provisioning:/etc/grafana/provisioning
    networks:
      - retentionai-network
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Alertmanager - Alert Management
  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: retentionai-alertmanager
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - ./config/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--log.level=info'
    networks:
      - retentionai-network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Node Exporter - System Metrics
  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: retentionai-node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
      - '--web.listen-address=0.0.0.0:9100'
    networks:
      - retentionai-network

  # Nginx - Reverse Proxy & Load Balancer
  nginx:
    image: nginx:1.24-alpine
    container_name: retentionai-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - /opt/retentionai/logs/nginx:/var/log/nginx
    networks:
      - retentionai-network
    depends_on:
      - retentionai-app
      - grafana
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Log Shipper
  promtail:
    image: grafana/promtail:2.8.0
    container_name: retentionai-promtail
    restart: unless-stopped
    volumes:
      - ./config/promtail.yml:/etc/promtail/config.yml
      - /opt/retentionai/logs:/var/log/retentionai:ro
      - /var/log:/var/log/host:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - retentionai-network

volumes:
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  alertmanager_data:
    driver: local

networks:
  retentionai-network:
    driver: bridge
    name: retentionai-network
EOF
    
    print_success "Created production Docker Compose configuration"
}

# Create Nginx configuration
create_nginx_config() {
    print_header "Creating Nginx Configuration"
    
    mkdir -p config
    
    cat > config/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/m;
    
    # Upstream definitions
    upstream retentionai_app {
        server retentionai-app:8501;
    }
    
    upstream grafana {
        server retentionai-grafana:3000;
    }
    
    upstream prometheus {
        server retentionai-prometheus:9090;
    }
    
    upstream alertmanager {
        server retentionai-alertmanager:9093;
    }
    
    # Main application
    server {
        listen 80;
        server_name localhost retentionai.local;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # Main application
        location / {
            proxy_pass http://retentionai_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Health checks
        location /health {
            proxy_pass http://retentionai-app:8001/health;
            proxy_set_header Host $host;
            access_log off;
        }
        
        # Metrics endpoint (internal only)
        location /metrics {
            proxy_pass http://retentionai-app:8000/metrics;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
        
        # Grafana dashboard
        location /grafana/ {
            rewrite ^/grafana(/.*)$ $1 break;
            proxy_pass http://grafana;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Prometheus (admin access only)
        location /prometheus/ {
            rewrite ^/prometheus(/.*)$ $1 break;
            proxy_pass http://prometheus;
            proxy_set_header Host $host;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
        
        # Alertmanager (admin access only)
        location /alertmanager/ {
            rewrite ^/alertmanager(/.*)$ $1 break;
            proxy_pass http://alertmanager;
            proxy_set_header Host $host;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
        }
    }
}
EOF
    
    print_success "Created Nginx configuration"
}

# Deploy application
deploy_application() {
    print_header "Deploying RetentionAI to Production"
    
    # Build application image
    print_status "Building application image..."
    docker build -t retentionai:latest .
    
    # Deploy with monitoring stack
    print_status "Starting production deployment..."
    docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to start..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    services=("retentionai-app" "retentionai-prometheus" "retentionai-grafana" "retentionai-nginx")
    
    for service in "${services[@]}"; do
        if docker ps | grep -q "$service"; then
            print_success "Service $service is running"
        else
            print_error "Service $service failed to start"
            docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" logs "$service"
        fi
    done
}

# Validate deployment
validate_deployment() {
    print_header "Validating Production Deployment"
    
    # Test endpoints
    endpoints=(
        "http://localhost/health:Health Check"
        "http://localhost:Application"
        "http://localhost:3000:Grafana"
        "http://localhost:9090:Prometheus"
        "http://localhost:9093:Alertmanager"
    )
    
    for endpoint in "${endpoints[@]}"; do
        url=$(echo $endpoint | cut -d: -f1)
        name=$(echo $endpoint | cut -d: -f2-)
        
        if curl -s -f "$url" > /dev/null; then
            print_success "$name is accessible at $url"
        else
            print_warning "$name may not be ready yet at $url"
        fi
    done
    
    # Check logs for errors
    print_status "Checking application logs for errors..."
    if docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" logs retentionai-app | grep -i error; then
        print_warning "Found errors in application logs"
    else
        print_success "No errors found in application logs"
    fi
}

# Setup monitoring
setup_monitoring() {
    print_header "Setting Up Production Monitoring"
    
    # Start monitoring stack
    ./scripts/monitoring.sh start
    
    # Wait for monitoring to be ready
    sleep 20
    
    # Import Grafana dashboard
    print_status "Setting up Grafana dashboard..."
    
    # Test Grafana API
    for i in {1..10}; do
        if curl -s -f -u admin:retentionai123 "http://localhost:3000/api/health" > /dev/null; then
            print_success "Grafana is ready"
            break
        else
            print_status "Waiting for Grafana to be ready... (attempt $i/10)"
            sleep 5
        fi
    done
}

# Create systemd service
create_systemd_service() {
    print_header "Creating Systemd Service"
    
    cat > /etc/systemd/system/retentionai.service << EOF
[Unit]
Description=RetentionAI Production Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/retentionai
ExecStart=/usr/local/bin/docker-compose -f docker-compose.production.yml -p $COMPOSE_PROJECT up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.production.yml -p $COMPOSE_PROJECT down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable retentionai
    
    print_success "Created and enabled systemd service"
}

# Setup backup cron job
setup_backup_cron() {
    print_header "Setting Up Automated Backups"
    
    # Create backup script
    cat > /opt/retentionai/scripts/backup.sh << 'EOF'
#!/bin/bash

# RetentionAI Automated Backup Script
BACKUP_DIR="/opt/retentionai/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR/database/$DATE"
mkdir -p "$BACKUP_DIR/models/$DATE"

# Backup database
cp /opt/retentionai/data/*.db "$BACKUP_DIR/database/$DATE/" 2>/dev/null || true

# Backup models
cp -r /opt/retentionai/data/models/* "$BACKUP_DIR/models/$DATE/" 2>/dev/null || true

# Backup monitoring data
docker run --rm \
    -v retentionai-production_prometheus_data:/data \
    -v "$BACKUP_DIR:/backup" \
    busybox tar czf "/backup/prometheus_$DATE.tar.gz" -C /data .

# Cleanup old backups
find "$BACKUP_DIR" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true

echo "Backup completed: $DATE"
EOF
    
    chmod +x /opt/retentionai/scripts/backup.sh
    
    # Add to crontab
    (crontab -l 2>/dev/null; echo "0 2 * * * /opt/retentionai/scripts/backup.sh >> /opt/retentionai/logs/backup.log 2>&1") | crontab -
    
    print_success "Setup automated daily backups at 2 AM"
}

# Print deployment summary
print_summary() {
    print_header "Deployment Summary"
    
    cat << EOF

🎯 RetentionAI Production Deployment Complete!

📊 Application URLs:
   Main App:      http://localhost/
   Grafana:       http://localhost:3000 (admin/retentionai123)
   Prometheus:    http://localhost:9090
   Alertmanager:  http://localhost:9093
   MLflow:        http://localhost:5000

🔍 Monitoring:
   Health:        http://localhost/health
   Metrics:       http://localhost/metrics
   System Status: ./scripts/monitoring.sh status

📁 Important Paths:
   Data:          /opt/retentionai/data
   Logs:          /opt/retentionai/logs  
   Backups:       /opt/retentionai/backups
   Config:        $(pwd)/config

🛠️  Management Commands:
   Service:       systemctl [start|stop|status] retentionai
   Monitoring:    ./scripts/monitoring.sh [start|stop|status]
   Backup:        /opt/retentionai/scripts/backup.sh
   Logs:          docker-compose logs -f [service]

🚀 Next Steps:
   1. Configure alerting (config/alertmanager.yml)
   2. Set up SSL certificates for HTTPS
   3. Configure external monitoring endpoints
   4. Review and adjust resource limits
   5. Set up log rotation and cleanup jobs

EOF
}

# Main deployment workflow
main() {
    case "${1:-}" in
        "dev"|"development")
            print_header "Development Deployment"
            check_requirements
            docker-compose -f docker-compose.monitoring.yml up -d
            print_success "Development environment started"
            ;;
        "prod"|"production")
            check_privileges "production"
            print_header "Production Deployment"
            check_requirements
            setup_directories
            create_production_env
            create_production_compose
            create_nginx_config
            deploy_application
            setup_monitoring
            create_systemd_service
            setup_backup_cron
            validate_deployment
            print_summary
            ;;
        "validate")
            print_header "Deployment Validation"
            validate_deployment
            ;;
        "status")
            print_header "Deployment Status"
            docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" ps
            ./scripts/monitoring.sh status
            ;;
        "stop")
            print_header "Stopping Production Deployment"
            docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" down
            ./scripts/monitoring.sh stop
            ;;
        "restart")
            print_header "Restarting Production Deployment"
            docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" restart
            ;;
        "logs")
            service="${2:-retentionai-app}"
            docker-compose -f docker-compose.production.yml -p "$COMPOSE_PROJECT" logs -f "$service"
            ;;
        "backup")
            print_header "Manual Backup"
            /opt/retentionai/scripts/backup.sh
            ;;
        *)
            echo "RetentionAI Production Deployment Script"
            echo ""
            echo "Usage: $0 {dev|prod|validate|status|stop|restart|logs|backup}"
            echo ""
            echo "Commands:"
            echo "  dev        Start development environment"
            echo "  prod       Deploy to production (requires root)"
            echo "  validate   Validate current deployment"
            echo "  status     Show deployment status"
            echo "  stop       Stop production deployment"
            echo "  restart    Restart production deployment"
            echo "  logs       Show service logs [service]"
            echo "  backup     Run manual backup"
            echo ""
            echo "Examples:"
            echo "  $0 dev                    # Start development"
            echo "  sudo $0 prod              # Deploy to production"
            echo "  $0 logs retentionai-app   # View app logs"
            echo "  $0 status                 # Check status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"