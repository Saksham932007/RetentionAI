# RetentionAI Deployment Guide

This directory contains all deployment configurations and scripts for RetentionAI across different environments.

## 🚀 Quick Start

### Development Environment
```bash
# Start the application
./deployment/deploy.sh up

# Access the application
open https://localhost
```

### Production Environment
```bash
# Deploy to production
./deployment/deploy.sh -e production up

# Check status
./deployment/deploy.sh -e production status
```

## 📁 File Structure

```
deployment/
├── docker-entrypoint.sh       # Container initialization script
├── deploy.sh                   # Main deployment script
├── nginx.conf                  # Nginx reverse proxy configuration
├── ssl/
│   ├── generate-certs.sh      # SSL certificate generation
│   ├── cert.pem               # SSL certificate (generated)
│   └── key.pem                # SSL private key (generated)
└── README.md                   # This file
```

## 🐳 Docker Configuration

### Multi-Stage Dockerfile

The `Dockerfile` uses multi-stage builds to create optimized images:

- **base**: Common Python environment
- **development**: Development tools and hot-reload
- **production**: Optimized for production deployment
- **training**: ML training environment with additional packages
- **api**: FastAPI service environment (future)

### Docker Compose Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| default | app, mlflow | Development |
| monitoring | + prometheus, grafana | Staging/Monitoring |
| production | + nginx, api | Full production |
| training | trainer | ML training jobs |
| api | api service | API-only deployment |
| cache | + redis | Caching layer |

## 🌍 Environments

### Development
- Single Streamlit app + MLflow
- Hot-reload enabled
- Debug logging
- Self-signed SSL certificates

### Staging
- Production-like setup
- Monitoring enabled
- Performance testing
- SSL certificates

### Production
- Full stack deployment
- Nginx reverse proxy
- SSL termination
- Rate limiting
- Health checks
- Monitoring and alerts

## 📋 Deployment Commands

### Basic Operations
```bash
# Start services
./deploy.sh up

# Stop services
./deploy.sh down

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs -f retentionai-app

# Check status
./deploy.sh status
```

### Environment-Specific
```bash
# Production deployment
./deploy.sh -e production up

# Training job
./deploy.sh -e training up

# Staging with monitoring
./deploy.sh -e staging up
```

### Maintenance
```bash
# Rebuild services
./deploy.sh rebuild

# Clean up resources
./deploy.sh clean

# Open shell in container
./deploy.sh shell retentionai-app
```

## 🔧 Configuration

### Environment Variables

Create environment-specific files:

**`.env.development`**
```bash
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///app/data/retentionai.db
MLFLOW_TRACKING_URI=http://mlflow:5000
MODELS_DIR=/app/models
WORKERS=1
```

**`.env.production`**
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///app/data/retentionai.db
MLFLOW_TRACKING_URI=http://mlflow:5000
MODELS_DIR=/app/models
WORKERS=4
```

### SSL Certificates

For development:
```bash
cd deployment/ssl
bash generate-certs.sh
```

For production, replace with CA-signed certificates:
```bash
cp your-cert.pem deployment/ssl/cert.pem
cp your-key.pem deployment/ssl/key.pem
```

## 🔍 Monitoring

### Health Checks

Services include built-in health checks:

- **Streamlit**: `http://localhost:8501/_stcore/health`
- **MLflow**: `http://localhost:5000/health`
- **API**: `http://localhost:8000/health` (when implemented)

### Monitoring Stack

- **Prometheus**: Metrics collection (`http://localhost:9090`)
- **Grafana**: Dashboards (`http://localhost:3000`)
- **Nginx**: Access logs and metrics

### Log Management

```bash
# Application logs
docker compose logs retentionai-app

# MLflow logs
docker compose logs mlflow

# All service logs
docker compose logs

# Follow logs in real-time
docker compose logs -f
```

## 🔒 Security

### Production Security Features

- Non-root container user
- SSL/TLS encryption
- Security headers (HSTS, CSP, etc.)
- Rate limiting
- Request size limits
- Attack pattern blocking

### Firewall Rules

```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8501/tcp  # Block direct Streamlit access
```

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   chmod +x deployment/deploy.sh
   chmod +x deployment/docker-entrypoint.sh
   ```

2. **SSL Certificate Errors**
   ```bash
   cd deployment/ssl
   rm cert.pem key.pem
   bash generate-certs.sh
   ```

3. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8501
   
   # Stop conflicting services
   docker compose down
   ```

4. **Database Issues**
   ```bash
   # Reset database
   rm data/retentionai.db
   docker compose restart retentionai-app
   ```

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG docker compose up

# Access container shell
docker compose exec retentionai-app bash

# Check container health
docker compose ps
docker compose top
```

## 📊 Performance

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Streamlit App | 0.5 CPU | 1GB | 100MB |
| MLflow | 0.2 CPU | 512MB | 1GB |
| Nginx | 0.1 CPU | 128MB | 10MB |
| Monitoring | 0.3 CPU | 1GB | 5GB |

### Scaling

```bash
# Scale Streamlit app
docker compose up --scale retentionai-app=3

# Load balancer will distribute traffic
```

## 🔄 CI/CD Integration

### GitHub Actions Integration

```yaml
# In .github/workflows/deploy.yml
- name: Deploy to production
  run: |
    ./deployment/deploy.sh -e production build
    ./deployment/deploy.sh -e production up
```

### Health Check Integration

```bash
# Wait for services to be healthy
./deploy.sh up
sleep 30
curl -f http://localhost/_stcore/health || exit 1
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Nginx Configuration Guide](https://nginx.org/en/docs/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [MLflow Deployment](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers)