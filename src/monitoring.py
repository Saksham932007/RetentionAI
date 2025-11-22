"""
Production monitoring and metrics collection for RetentionAI.

This module provides comprehensive monitoring capabilities including:
- Application performance metrics
- ML model performance tracking
- Business metrics collection
- Health checks and alerting
- Custom Prometheus metrics
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from datetime import datetime, timedelta
import threading
import sqlite3
import json
from pathlib import Path

# Prometheus metrics (optional dependency)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
    from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Centralized metrics collection for RetentionAI."""
    
    def __init__(self, enable_prometheus: bool = True, metrics_port: int = 8000):
        """
        Initialize the metrics collector.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
            metrics_port: Port for Prometheus metrics endpoint
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_port = metrics_port
        self._metrics = {}
        self._custom_metrics = {}
        
        if self.enable_prometheus:
            self._setup_prometheus_metrics()
            self._start_metrics_server()
        
        # In-memory metrics storage for fallback
        self._in_memory_metrics = {
            'counters': {},
            'gauges': {},
            'histograms': {},
            'summaries': {}
        }
        
        logger.info(f"Metrics collector initialized (Prometheus: {self.enable_prometheus})")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Application metrics
        self._metrics['requests_total'] = Counter(
            'retentionai_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self._metrics['request_duration'] = Histogram(
            'retentionai_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self._metrics['active_connections'] = Gauge(
            'retentionai_active_connections',
            'Number of active connections'
        )
        
        # ML model metrics
        self._metrics['predictions_total'] = Counter(
            'retentionai_predictions_total',
            'Total number of predictions made',
            ['model_version', 'prediction_type']
        )
        
        self._metrics['prediction_duration'] = Histogram(
            'retentionai_prediction_duration_seconds',
            'Prediction duration in seconds',
            ['model_version']
        )
        
        self._metrics['model_accuracy'] = Gauge(
            'retentionai_model_accuracy',
            'Current model accuracy',
            ['model_version']
        )
        
        self._metrics['model_auc'] = Gauge(
            'retentionai_model_auc',
            'Current model AUC score',
            ['model_version']
        )
        
        self._metrics['churn_prediction_rate'] = Gauge(
            'retentionai_churn_prediction_rate',
            'Current churn prediction rate'
        )
        
        # Data metrics
        self._metrics['data_quality_score'] = Gauge(
            'retentionai_data_quality_score',
            'Data quality score (0-1)'
        )
        
        self._metrics['data_drift_score'] = Gauge(
            'retentionai_data_drift_score',
            'Data drift score (0-1)'
        )
        
        self._metrics['training_jobs_total'] = Counter(
            'retentionai_training_jobs_total',
            'Total number of training jobs',
            ['status']
        )
        
        # Database metrics
        self._metrics['database_connections'] = Gauge(
            'retentionai_database_connections',
            'Number of active database connections'
        )
        
        self._metrics['database_query_duration'] = Histogram(
            'retentionai_database_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type']
        )
        
        # System metrics
        self._metrics['memory_usage_bytes'] = Gauge(
            'retentionai_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self._metrics['cpu_usage_percent'] = Gauge(
            'retentionai_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Business metrics
        self._metrics['customers_processed_total'] = Counter(
            'retentionai_customers_processed_total',
            'Total number of customers processed'
        )
        
        self._metrics['revenue_impact'] = Gauge(
            'retentionai_revenue_impact',
            'Estimated revenue impact from predictions'
        )
        
        # Application info
        self._metrics['app_info'] = Info(
            'retentionai_app_info',
            'Application information'
        )
        
        # Set application info
        self._metrics['app_info'].info({
            'version': '1.0.0',
            'environment': 'production',
            'build_date': datetime.now().isoformat()
        })
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server."""
        if not self.enable_prometheus:
            return
        
        try:
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            self.enable_prometheus = False
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1):
        """Increment a counter metric."""
        labels = labels or {}
        
        if self.enable_prometheus and name in self._metrics:
            if labels:
                self._metrics[name].labels(**labels).inc(value)
            else:
                self._metrics[name].inc(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}_{hash(frozenset(labels.items()) if labels else frozenset())}"
            self._in_memory_metrics['counters'][key] = \
                self._in_memory_metrics['counters'].get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        labels = labels or {}
        
        if self.enable_prometheus and name in self._metrics:
            if labels:
                self._metrics[name].labels(**labels).set(value)
            else:
                self._metrics[name].set(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}_{hash(frozenset(labels.items()) if labels else frozenset())}"
            self._in_memory_metrics['gauges'][key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric value."""
        labels = labels or {}
        
        if self.enable_prometheus and name in self._metrics:
            if labels:
                self._metrics[name].labels(**labels).observe(value)
            else:
                self._metrics[name].observe(value)
        else:
            # Fallback to in-memory storage
            key = f"{name}_{hash(frozenset(labels.items()) if labels else frozenset())}"
            if key not in self._in_memory_metrics['histograms']:
                self._in_memory_metrics['histograms'][key] = []
            self._in_memory_metrics['histograms'][key].append(value)
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track HTTP request metrics."""
        # Request counter
        self.increment_counter('requests_total', {
            'method': method,
            'endpoint': endpoint,
            'status': str(status_code)
        })
        
        # Request duration
        self.observe_histogram('request_duration', duration, {
            'method': method,
            'endpoint': endpoint
        })
    
    def track_prediction(self, model_version: str, prediction_type: str, duration: float, 
                        prediction_value: float = None):
        """Track ML prediction metrics."""
        # Prediction counter
        self.increment_counter('predictions_total', {
            'model_version': model_version,
            'prediction_type': prediction_type
        })
        
        # Prediction duration
        self.observe_histogram('prediction_duration', duration, {
            'model_version': model_version
        })
        
        # Track churn rate if it's a churn prediction
        if prediction_type == 'churn' and prediction_value is not None:
            current_rate = self.get_current_churn_rate()
            new_rate = (current_rate * 0.9 + prediction_value * 0.1)  # Moving average
            self.set_gauge('churn_prediction_rate', new_rate)
    
    def track_model_performance(self, model_version: str, accuracy: float, auc: float):
        """Track model performance metrics."""
        self.set_gauge('model_accuracy', accuracy, {'model_version': model_version})
        self.set_gauge('model_auc', auc, {'model_version': model_version})
    
    def track_training_job(self, status: str):
        """Track training job metrics."""
        self.increment_counter('training_jobs_total', {'status': status})
    
    def track_data_quality(self, quality_score: float, drift_score: float = None):
        """Track data quality metrics."""
        self.set_gauge('data_quality_score', quality_score)
        if drift_score is not None:
            self.set_gauge('data_drift_score', drift_score)
    
    def track_database_operation(self, query_type: str, duration: float):
        """Track database operation metrics."""
        self.observe_histogram('database_query_duration', duration, {
            'query_type': query_type
        })
    
    def track_system_resources(self, memory_bytes: int, cpu_percent: float):
        """Track system resource metrics."""
        self.set_gauge('memory_usage_bytes', memory_bytes)
        self.set_gauge('cpu_usage_percent', cpu_percent)
    
    def track_business_impact(self, customers_processed: int, revenue_impact: float):
        """Track business metrics."""
        self.increment_counter('customers_processed_total', value=customers_processed)
        self.set_gauge('revenue_impact', revenue_impact)
    
    def get_current_churn_rate(self) -> float:
        """Get current churn prediction rate."""
        if self.enable_prometheus:
            # This would typically query the Prometheus metric
            return 0.25  # Placeholder
        else:
            return self._in_memory_metrics['gauges'].get('churn_prediction_rate_', 0.25)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        if self.enable_prometheus:
            # In production, this would query Prometheus
            return {
                'prometheus_enabled': True,
                'metrics_port': self.metrics_port,
                'total_metrics': len(self._metrics)
            }
        else:
            return {
                'prometheus_enabled': False,
                'in_memory_metrics': {
                    'counters': len(self._in_memory_metrics['counters']),
                    'gauges': len(self._in_memory_metrics['gauges']),
                    'histograms': len(self._in_memory_metrics['histograms'])
                }
            }


# Global metrics collector instance
metrics_collector = MetricsCollector()


def monitor_performance(metric_name: str = None):
    """
    Decorator to monitor function performance.
    
    Args:
        metric_name: Custom metric name, defaults to function name
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = metric_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Track successful execution
                metrics_collector.observe_histogram(
                    'function_duration_seconds',
                    duration,
                    {'function': function_name, 'status': 'success'}
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Track failed execution
                metrics_collector.observe_histogram(
                    'function_duration_seconds',
                    duration,
                    {'function': function_name, 'status': 'error'}
                )
                
                metrics_collector.increment_counter(
                    'function_errors_total',
                    {'function': function_name, 'error_type': type(e).__name__}
                )
                
                raise
        
        return wrapper
    return decorator


class HealthChecker:
    """Health check system for RetentionAI components."""
    
    def __init__(self, database_path: str = "data/retentionai.db"):
        """Initialize health checker."""
        self.database_path = database_path
        self.checks = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check('database', self._check_database)
        self.register_check('filesystem', self._check_filesystem)
        self.register_check('memory', self._check_memory)
        self.register_check('model_availability', self._check_model_availability)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a new health check."""
        self.checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Test database connection
            with sqlite3.connect(self.database_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
            
            duration = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time_ms': round(duration * 1000, 2),
                'table_count': table_count,
                'database_size_mb': self._get_database_size()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': None
            }
    
    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem availability and disk space."""
        try:
            import shutil
            
            # Check disk usage
            total, used, free = shutil.disk_usage('.')
            
            free_gb = free // (2**30)
            total_gb = total // (2**30)
            usage_percent = (used / total) * 100
            
            status = 'healthy' if free_gb > 1 and usage_percent < 90 else 'warning'
            if free_gb < 0.5 or usage_percent > 95:
                status = 'unhealthy'
            
            return {
                'status': status,
                'free_space_gb': free_gb,
                'total_space_gb': total_gb,
                'usage_percent': round(usage_percent, 1)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            status = 'healthy' if memory.percent < 85 else 'warning'
            if memory.percent > 95:
                status = 'unhealthy'
            
            return {
                'status': status,
                'usage_percent': round(memory.percent, 1),
                'available_gb': round(memory.available / (2**30), 2),
                'total_gb': round(memory.total / (2**30), 2)
            }
            
        except ImportError:
            # Fallback if psutil not available
            return {
                'status': 'unknown',
                'error': 'psutil not available for memory monitoring'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _check_model_availability(self) -> Dict[str, Any]:
        """Check if ML models are available and loaded."""
        try:
            models_dir = Path('models')
            
            if not models_dir.exists():
                return {
                    'status': 'warning',
                    'error': 'Models directory does not exist'
                }
            
            # Check for model files
            model_files = list(models_dir.glob('**/*.pkl')) + list(models_dir.glob('**/*.joblib'))
            
            status = 'healthy' if model_files else 'warning'
            
            return {
                'status': status,
                'available_models': len(model_files),
                'models_directory_exists': True
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def _get_database_size(self) -> float:
        """Get database file size in MB."""
        try:
            db_path = Path(self.database_path)
            if db_path.exists():
                size_bytes = db_path.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except:
            return 0.0
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = 'healthy'
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                # Determine overall status
                if result['status'] == 'unhealthy':
                    overall_status = 'unhealthy'
                elif result['status'] == 'warning' and overall_status == 'healthy':
                    overall_status = 'warning'
                    
            except Exception as e:
                results[check_name] = {
                    'status': 'unhealthy',
                    'error': f"Health check failed: {str(e)}"
                }
                overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': results
        }
    
    def run_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if check_name not in self.checks:
            return {
                'status': 'unhealthy',
                'error': f"Unknown health check: {check_name}"
            }
        
        try:
            return self.checks[check_name]()
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f"Health check failed: {str(e)}"
            }


# Global health checker instance
health_checker = HealthChecker()


class AlertManager:
    """Alert management system for RetentionAI."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize alert manager."""
        self.config = config or {}
        self.alert_rules = {}
        self.active_alerts = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        # Model performance alerts
        self.add_rule(
            'model_accuracy_low',
            lambda metrics: metrics.get('model_accuracy', 1.0) < 0.8,
            severity='critical',
            message='Model accuracy below 80% threshold'
        )
        
        self.add_rule(
            'high_churn_rate',
            lambda metrics: metrics.get('churn_prediction_rate', 0.0) > 0.5,
            severity='warning',
            message='Churn prediction rate above 50%'
        )
        
        # System health alerts
        self.add_rule(
            'database_slow',
            lambda metrics: metrics.get('database_response_time_ms', 0) > 1000,
            severity='warning',
            message='Database response time above 1 second'
        )
        
        self.add_rule(
            'memory_usage_high',
            lambda metrics: metrics.get('memory_usage_percent', 0) > 90,
            severity='critical',
            message='Memory usage above 90%'
        )
    
    def add_rule(self, name: str, condition: Callable, severity: str = 'warning', 
                 message: str = '', cooldown: int = 300):
        """Add an alert rule."""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'message': message,
            'cooldown': cooldown,
            'last_triggered': None
        }
        logger.debug(f"Added alert rule: {name}")
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check all alert rules against current metrics."""
        triggered_alerts = []
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check if condition is met
                if rule['condition'](metrics):
                    # Check cooldown period
                    last_triggered = rule.get('last_triggered')
                    if last_triggered:
                        time_since_last = (current_time - last_triggered).total_seconds()
                        if time_since_last < rule['cooldown']:
                            continue
                    
                    # Trigger alert
                    alert = {
                        'name': rule_name,
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': current_time.isoformat(),
                        'metrics_snapshot': metrics.copy()
                    }
                    
                    triggered_alerts.append(alert)
                    self.active_alerts[rule_name] = alert
                    rule['last_triggered'] = current_time
                    
                    logger.warning(f"Alert triggered: {rule_name} - {rule['message']}")
                
                # Check if alert should be resolved
                elif rule_name in self.active_alerts:
                    resolved_alert = self.active_alerts.pop(rule_name)
                    logger.info(f"Alert resolved: {rule_name}")
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        return list(self.active_alerts.values())
    
    def send_notification(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder for integration)."""
        # This would integrate with notification systems like:
        # - Slack
        # - PagerDuty
        # - Email
        # - Discord
        # - Teams
        
        logger.info(f"ALERT NOTIFICATION: {alert['severity'].upper()} - {alert['message']}")
        
        # For now, just log the alert
        print(f"🚨 ALERT: {alert['name']}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Message: {alert['message']}")
        print(f"   Time: {alert['timestamp']}")


# Global alert manager instance
alert_manager = AlertManager()


def setup_monitoring(enable_prometheus: bool = True, metrics_port: int = 8000) -> None:
    """Setup monitoring system for RetentionAI."""
    global metrics_collector, health_checker, alert_manager
    
    # Initialize components
    metrics_collector = MetricsCollector(enable_prometheus, metrics_port)
    health_checker = HealthChecker()
    alert_manager = AlertManager()
    
    logger.info("Monitoring system initialized")


def get_monitoring_status() -> Dict[str, Any]:
    """Get comprehensive monitoring status."""
    return {
        'metrics': metrics_collector.get_metrics_summary(),
        'health': health_checker.run_all_checks(),
        'alerts': {
            'active_count': len(alert_manager.get_active_alerts()),
            'active_alerts': alert_manager.get_active_alerts()
        },
        'monitoring_timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Test monitoring system
    setup_monitoring(enable_prometheus=False)  # Disable for testing
    
    # Test metrics
    metrics_collector.track_prediction('v1.0.0', 'churn', 0.15, 0.3)
    metrics_collector.track_model_performance('v1.0.0', 0.87, 0.91)
    
    # Test health checks
    health_status = health_checker.run_all_checks()
    print("Health Status:", json.dumps(health_status, indent=2))
    
    # Test monitoring status
    status = get_monitoring_status()
    print("Monitoring Status:", json.dumps(status, indent=2))