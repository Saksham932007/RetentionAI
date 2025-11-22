"""
Health check endpoints for RetentionAI Streamlit application.

This module provides FastAPI endpoints for comprehensive health monitoring
that can be integrated with the main Streamlit application.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import time
import logging
from datetime import datetime
from typing import Dict, Any
import asyncio
import uvicorn
import threading

from src.monitoring import health_checker, metrics_collector, get_monitoring_status

logger = logging.getLogger(__name__)

# Create FastAPI app for health endpoints
health_app = FastAPI(
    title="RetentionAI Health API",
    description="Health monitoring endpoints for RetentionAI application",
    version="1.0.0"
)


@health_app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns overall application health status.
    """
    try:
        start_time = time.time()
        health_status = health_checker.run_all_checks()
        duration = time.time() - start_time
        
        # Track health check metrics
        metrics_collector.observe_histogram('health_check_duration_seconds', duration)
        
        # Determine HTTP status code based on health
        if health_status['overall_status'] == 'healthy':
            status_code = status.HTTP_200_OK
        elif health_status['overall_status'] == 'warning':
            status_code = status.HTTP_200_OK  # Warning still returns 200
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content=health_status,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@health_app.get("/health/{component}", response_model=Dict[str, Any])
async def component_health_check(component: str):
    """
    Health check for specific component.
    
    Args:
        component: Component name (database, filesystem, memory, model_availability)
    """
    try:
        result = health_checker.run_check(component)
        
        if result['status'] == 'healthy':
            status_code = status.HTTP_200_OK
        elif result['status'] == 'warning':
            status_code = status.HTTP_200_OK
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            content={
                'component': component,
                'timestamp': datetime.now().isoformat(),
                **result
            },
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Component health check failed for {component}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component health check failed: {str(e)}"
        )


@health_app.get("/ready", response_model=Dict[str, Any])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    Returns 200 if application is ready to serve requests.
    """
    try:
        # Check critical components for readiness
        critical_checks = ['database', 'model_availability']
        
        for check_name in critical_checks:
            result = health_checker.run_check(check_name)
            if result['status'] == 'unhealthy':
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Application not ready: {check_name} is unhealthy"
                )
        
        return JSONResponse(
            content={
                'status': 'ready',
                'timestamp': datetime.now().isoformat(),
                'message': 'Application is ready to serve requests'
            },
            status_code=status.HTTP_200_OK
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@health_app.get("/live", response_model=Dict[str, Any])
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    Returns 200 if application is alive (basic functionality).
    """
    try:
        return JSONResponse(
            content={
                'status': 'alive',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - start_time
            },
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Liveness check failed: {str(e)}"
        )


@health_app.get("/metrics/summary", response_model=Dict[str, Any])
async def metrics_summary():
    """
    Get metrics summary endpoint.
    Returns current application metrics.
    """
    try:
        monitoring_status = get_monitoring_status()
        
        return JSONResponse(
            content=monitoring_status,
            status_code=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics summary failed: {str(e)}"
        )


@health_app.get("/version", response_model=Dict[str, Any])
async def version_info():
    """
    Application version and build information.
    """
    return JSONResponse(
        content={
            'application': 'RetentionAI',
            'version': '1.0.0',
            'build_date': '2024-01-20',
            'environment': 'production',
            'python_version': '3.10+',
            'features': [
                'churn_prediction',
                'model_training',
                'data_processing',
                'monitoring',
                'alerting'
            ]
        },
        status_code=status.HTTP_200_OK
    )


# Global start time for uptime calculation
start_time = time.time()


class HealthServer:
    """Health check server manager."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        """
        Initialize health server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
    
    def start_server(self):
        """Start the health check server in a separate thread."""
        def run_server():
            try:
                config = uvicorn.Config(
                    health_app,
                    host=self.host,
                    port=self.port,
                    log_level="info"
                )
                self.server = uvicorn.Server(config)
                asyncio.run(self.server.serve())
            except Exception as e:
                logger.error(f"Health server failed: {e}")
        
        if not self.running:
            self.thread = threading.Thread(target=run_server, daemon=True)
            self.thread.start()
            self.running = True
            logger.info(f"Health check server started on {self.host}:{self.port}")
    
    def stop_server(self):
        """Stop the health check server."""
        if self.server and self.running:
            self.server.shutdown()
            self.running = False
            logger.info("Health check server stopped")


# Global health server instance
health_server = HealthServer()


def start_health_server(host: str = "0.0.0.0", port: int = 8001):
    """
    Start health check server.
    
    Args:
        host: Server host
        port: Server port
    """
    global health_server
    health_server = HealthServer(host, port)
    health_server.start_server()


def stop_health_server():
    """Stop health check server."""
    global health_server
    if health_server:
        health_server.stop_server()


if __name__ == "__main__":
    # Run health server standalone for testing
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
    
    uvicorn.run(health_app, host=host, port=port)