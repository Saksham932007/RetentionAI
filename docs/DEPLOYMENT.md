# Production Deployment Guide for RetentionAI

## Overview

This guide provides comprehensive instructions for deploying RetentionAI to production with full monitoring, security, and scalability features.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Production Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    Nginx    │  │  RetentionAI │  │   MLflow    │            │
│  │ Load Balancer│  │     App     │  │   Server    │            │
│  │   Port 80   │  │  Port 8501  │  │  Port 5000  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│          │               │               │                     │
│          └───────────────┼───────────────┘                     │
│                          │                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Monitoring Stack                        │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Prometheus  │  │   Grafana   │  │ Alertmanager│     │   │
│  │  │  Port 9090  │  │  Port 3000  │  │  Port 9093  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │Node Exporter│  │   Promtail  │  │    Loki     │     │   │
│  │  │  Port 9100  │  │             │  │  Port 3100  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, or similar)
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM recommended  
- **Storage**: 50GB+ available disk space
- **Network**: Stable internet connection

### Software Dependencies

```bash
# Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Additional tools
sudo apt-get update
sudo apt-get install -y curl wget git apache2-utils bc
```

## Deployment Options

### Option 1: Docker Compose Deployment (Recommended)

#### 1. Clone and Prepare

```bash
git clone https://github.com/your-repo/RetentionAI.git
cd RetentionAI
```

#### 2. Deploy to Production

```bash
# Deploy with full production setup
sudo ./scripts/deploy.sh prod
```

This command will:
- Create production directories (`/opt/retentionai/`)
- Generate production environment configuration
- Build and deploy all containers
- Set up monitoring stack
- Configure Nginx reverse proxy
- Create systemd service
- Set up automated backups
- Validate deployment

#### 3. Validate Deployment

```bash
# Run comprehensive validation
./scripts/validate.sh

# Check service status
./scripts/deploy.sh status
```

### Option 2: Kubernetes Deployment

#### 1. Apply Kubernetes Manifests

```bash
# Create namespace and deploy
kubectl apply -f k8s/production.yaml

# Verify deployment
kubectl get pods -n retentionai
kubectl get services -n retentionai
```

#### 2. Configure Ingress

```bash
# Update ingress with your domain
kubectl edit ingress retentionai-ingress -n retentionai
```

## Configuration

### Environment Variables

Key production environment variables in `.env.production`:

```bash
# Application
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=sqlite:///data/retentionai.db

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=secure_password_here

# Security
SECRET_KEY=your_32_character_secret_key_here
ALLOWED_HOSTS=your-domain.com,localhost

# Performance
WORKERS=4
MAX_CONNECTIONS=100
```

### Security Configuration

#### 1. Change Default Passwords

```bash
# Update Grafana admin password
export GRAFANA_ADMIN_PASSWORD="your_secure_password"

# Update secret key
openssl rand -hex 32
```

#### 2. Configure SSL/TLS

For HTTPS, add SSL certificates to Nginx configuration:

```bash
# Generate self-signed certificate (for testing)
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/retentionai.key \
  -out /etc/ssl/certs/retentionai.crt

# Update nginx.conf with SSL configuration
```

#### 3. Network Security

```bash
# Configure firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw --force enable

# Restrict access to monitoring ports
sudo ufw deny 9090/tcp  # Prometheus
sudo ufw deny 3000/tcp  # Grafana direct access
sudo ufw deny 9093/tcp  # Alertmanager
```

## Monitoring Setup

### 1. Access Monitoring Dashboards

```bash
# Main application
http://your-domain/

# Grafana dashboards (via Nginx proxy)
http://your-domain/grafana/

# Or direct access (if firewall allows)
http://your-domain:3000
```

### 2. Configure Alerts

Update `config/alertmanager.yml` with your notification preferences:

```yaml
# Email notifications
receivers:
  - name: 'critical-alerts'
    email_configs:
      - to: 'alerts@yourcompany.com'
        smtp_smarthost: 'smtp.yourcompany.com:587'
        smtp_from: 'retentionai@yourcompany.com'

# Slack notifications  
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#retentionai-alerts'
```

### 3. Import Grafana Dashboards

Grafana dashboards are automatically provisioned. Access them at:
- Application Overview Dashboard
- Infrastructure Monitoring Dashboard  
- ML Model Performance Dashboard

## Performance Tuning

### 1. Resource Limits

Update `docker-compose.production.yml` with appropriate resource limits:

```yaml
services:
  retentionai-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 2. Database Optimization

```bash
# For high-traffic deployments, consider PostgreSQL
DATABASE_URL=postgresql://username:password@host:5432/retentionai
```

### 3. Caching Configuration

```bash
# Enable Redis for caching (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600
```

## Backup and Recovery

### 1. Automated Backups

Automated backups run daily at 2 AM via cron job:

```bash
# Check backup status
ls -la /opt/retentionai/backups/

# Manual backup
/opt/retentionai/scripts/backup.sh

# Restore from backup
docker cp backup_file.tar.gz retentionai-app:/app/data/
```

### 2. Database Backup

```bash
# SQLite backup
docker exec retentionai-app sqlite3 /app/data/retentionai.db ".backup /app/data/backup.db"

# PostgreSQL backup (if using PostgreSQL)
pg_dump retentionai > backup.sql
```

## Load Testing

### 1. Basic Load Test

```bash
# Quick performance test
./scripts/load_test.sh quick

# Full load test with 20 concurrent users for 2 minutes
./scripts/load_test.sh full http://localhost 20 120
```

### 2. Stress Testing

```bash
# Stress test under high load
./scripts/load_test.sh stress

# Memory leak detection
./scripts/load_test.sh memory
```

## Troubleshooting

### 1. Check Service Health

```bash
# Overall system status
./scripts/deploy.sh status

# Individual service logs
docker-compose -f docker-compose.production.yml logs retentionai-app
docker-compose -f docker-compose.production.yml logs retentionai-prometheus
```

### 2. Common Issues

#### Application Won't Start
```bash
# Check logs for errors
docker logs retentionai-app

# Verify environment configuration
docker exec retentionai-app env | grep -E "(DATABASE|MLFLOW)"

# Check file permissions
ls -la /opt/retentionai/data/
```

#### Monitoring Not Working
```bash
# Restart monitoring stack
./scripts/monitoring.sh restart

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Monitor system resources
htop
iostat -x 1

# Check database performance
./scripts/validate.sh
```

## Scaling

### 1. Horizontal Scaling

For Kubernetes deployments, enable horizontal pod autoscaling:

```bash
kubectl autoscale deployment retentionai-app --cpu-percent=70 --min=2 --max=10 -n retentionai
```

### 2. Database Scaling

For high-traffic scenarios, migrate to PostgreSQL:

```bash
# Update environment
DATABASE_URL=postgresql://user:pass@host:5432/retentionai

# Migrate data
python scripts/migrate_to_postgresql.py
```

### 3. Load Balancing

Configure multiple application instances:

```yaml
services:
  retentionai-app:
    deploy:
      replicas: 3
```

## Security Best Practices

### 1. Regular Updates

```bash
# Update container images
docker-compose pull
docker-compose up -d

# Update system packages
sudo apt update && sudo apt upgrade -y
```

### 2. Access Control

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash retentionai
sudo usermod -aG docker retentionai

# Set proper file permissions
sudo chown -R retentionai:retentionai /opt/retentionai/
chmod 755 /opt/retentionai/scripts/*.sh
```

### 3. Network Security

```bash
# Configure Docker network isolation
docker network create --driver bridge retentionai-network

# Enable audit logging
echo "audit_log_enabled=1" >> /etc/mysql/mysql.conf.d/mysqld.cnf
```

## Maintenance

### 1. Regular Tasks

```bash
# Weekly maintenance script
#!/bin/bash
# - Check disk space
# - Rotate logs  
# - Update containers
# - Run validation tests
```

### 2. Health Monitoring

```bash
# Set up monitoring checks
./scripts/validate.sh  # Run weekly
./scripts/load_test.sh quick  # Run monthly
```

### 3. Disaster Recovery

```bash
# Create recovery procedures
# - Document backup locations
# - Test restore procedures
# - Document manual failover steps
```

## Support and Troubleshooting

### 1. Log Locations

```bash
# Application logs
/opt/retentionai/logs/application/

# Container logs
docker-compose logs [service_name]

# System logs
/var/log/retentionai/
```

### 2. Monitoring URLs

- **Application**: http://your-domain/
- **Health Check**: http://your-domain/health
- **Grafana**: http://your-domain:3000
- **Prometheus**: http://your-domain:9090
- **Alertmanager**: http://your-domain:9093

### 3. Emergency Procedures

```bash
# Emergency stop
sudo systemctl stop retentionai

# Emergency restart
sudo systemctl restart retentionai

# Rollback deployment
docker-compose down && docker-compose up -d
```

## Performance Benchmarks

Expected performance metrics for a standard deployment:

- **Response Time**: < 500ms (95th percentile)
- **Throughput**: 100+ requests/second
- **Memory Usage**: < 2GB under normal load
- **CPU Usage**: < 50% under normal load
- **Uptime**: > 99.9% availability

## Conclusion

This production deployment provides a robust, scalable, and monitored RetentionAI application with comprehensive observability, security, and operational features. Regular monitoring and maintenance ensure optimal performance and reliability.