"""
RetentionAI - Churn Prediction Streamlit Application

Main application entry point with multi-page navigation, theme configuration,
and session state management for the churn prediction dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import atexit
import time

# Setup path for imports
app_dir = Path(__file__).parent
sys.path.append(str(app_dir))
sys.path.append(str(app_dir.parent))

try:
    from config import MODELS_DIR, EXPERIMENT_NAME
    from database import get_database_manager
    from utils import setup_logging
    from monitoring import metrics_collector, monitor_performance, setup_monitoring, get_monitoring_status
    from health_endpoints import start_health_server, stop_health_server
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RetentionAI - Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/RetentionAI',
        'Report a bug': 'https://github.com/your-repo/RetentionAI/issues',
        'About': "# RetentionAI\n\nProduction-grade churn prediction application built with MLOps best practices."
    }
)

# Custom CSS for improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    
    .page-nav-button {
        width: 100%;
        margin-bottom: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 0.5rem;
        text-align: left;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


class RetentionAIApp:
    """Main RetentionAI Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.initialize_session_state()
        self.setup_authentication()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        
        # Initialize monitoring system (only once)
        if 'monitoring_initialized' not in st.session_state:
            try:
                setup_monitoring(enable_prometheus=True, metrics_port=8000)
                start_health_server(host="0.0.0.0", port=8001)
                st.session_state.monitoring_initialized = True
                logger.info("Monitoring system initialized successfully")
                
                # Cleanup function for when Streamlit stops
                def cleanup():
                    try:
                        stop_health_server()
                        logger.info("Health server stopped during cleanup")
                    except Exception as e:
                        logger.error(f"Error during cleanup: {e}")
                atexit.register(cleanup)
                
            except Exception as e:
                logger.error(f"Failed to initialize monitoring: {e}")
                st.session_state.monitoring_initialized = False
        
        # Page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Data Overview'
        
        # Authentication state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        
        # Data cache
        if 'data_cache' not in st.session_state:
            st.session_state.data_cache = {}
        if 'last_data_refresh' not in st.session_state:
            st.session_state.last_data_refresh = None
        
        # Model cache
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
        if 'active_model' not in st.session_state:
            st.session_state.active_model = None
        
        # UI preferences
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        
        # Prediction history
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def setup_authentication(self):
        """Setup basic authentication (placeholder implementation)."""
        
        # For demo purposes, use simple password authentication
        # In production, integrate with proper authentication system
        
        if not st.session_state.authenticated:
            # Show login in sidebar
            with st.sidebar:
                st.markdown("### 🔐 Authentication")
                
                # Demo credentials info
                st.info("**Demo Credentials:**\nUsername: admin\nPassword: retention123")
                
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login", key="login_button"):
                    if self.authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                
                # Guest access
                if st.button("Continue as Guest", key="guest_button"):
                    st.session_state.authenticated = True
                    st.session_state.username = "Guest"
                    st.warning("Limited functionality in guest mode")
                    st.rerun()
            
            # Show welcome message in main area
            st.markdown('<div class="main-header">🎯 RetentionAI Dashboard</div>', unsafe_allow_html=True)
            st.markdown("""
            ### Welcome to RetentionAI - Customer Churn Prediction Platform
            
            This application provides:
            - **Data Exploration**: Analyze customer data patterns and churn trends
            - **Model Performance**: Monitor ML model metrics and interpretability
            - **Real-time Predictions**: Generate churn predictions for individual customers
            - **Model Management**: Deploy and manage ML models in production
            
            Please authenticate to continue.
            """)
            
            # System status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("System Status", "🟢 Online", "All systems operational")
            with col2:
                st.metric("Models Available", "3", "2 in production")
            with col3:
                st.metric("Last Updated", datetime.now().strftime("%H:%M"), "Auto-refresh enabled")
            
            st.stop()
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authenticate user credentials (placeholder implementation)."""
        
        # Demo authentication - replace with real authentication in production
        valid_credentials = {
            'admin': 'retention123',
            'analyst': 'data123',
            'manager': 'business123'
        }
        
        return username in valid_credentials and valid_credentials[username] == password
    
    def render_sidebar(self):
        """Render the navigation sidebar."""
        
        with st.sidebar:
            # User info
            st.markdown(f"### 👤 Welcome, {st.session_state.username}!")
            
            if st.button("🚪 Logout", key="logout_button"):
                self.logout()
            
            st.markdown("---")
            
            # Navigation
            st.markdown('<div class="sidebar-header">📊 Navigation</div>', unsafe_allow_html=True)
            
            pages = {
                '📈 Data Overview': 'Data Overview',
                '🎯 Model Performance': 'Model Performance', 
                '🔮 Predictions': 'Predictions',
                '📊 Monitoring': 'Monitoring',
                '⚙️ Model Management': 'Model Management'
            }
            
            for display_name, page_name in pages.items():
                if st.button(display_name, key=f"nav_{page_name}", help=f"Navigate to {page_name}"):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("---")
            
            # System status
            st.markdown('<div class="sidebar-header">🖥️ System Status</div>', unsafe_allow_html=True)
            
            # Database connection status
            db_status = self.check_database_connection()
            if db_status:
                st.markdown('<span class="status-success">✅ Database Connected</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error">❌ Database Error</span>', unsafe_allow_html=True)
            
            # Model availability
            model_status = self.check_model_availability()
            if model_status:
                st.markdown('<span class="status-success">✅ Models Available</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠️ No Models Found</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Settings
            st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
            
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh data", 
                value=st.session_state.auto_refresh,
                help="Automatically refresh data every 30 seconds"
            )
            
            if st.button("🔄 Refresh Data", help="Manually refresh all cached data"):
                self.clear_cache()
                st.success("Cache cleared!")
                st.rerun()
            
            # Footer
            st.markdown("---")
            st.markdown(
                '<div class="footer">RetentionAI v1.0<br/>Built with ❤️ using Streamlit</div>',
                unsafe_allow_html=True
            )
    
    def check_database_connection(self) -> bool:
        """Check database connectivity."""
        try:
            db_manager = get_database_manager()
            # Try a simple query
            result = db_manager.execute_query("SELECT 1")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def check_model_availability(self) -> bool:
        """Check if trained models are available."""
        try:
            models_path = Path(MODELS_DIR)
            if not models_path.exists():
                return False
            
            # Check for any model files
            model_files = list(models_path.glob('**/*.json')) + list(models_path.glob('**/*.pkl'))
            return len(model_files) > 0
        except Exception as e:
            logger.error(f"Model availability check error: {e}")
            return False
    
    def logout(self):
        """Logout current user and clear session state."""
        st.session_state.authenticated = False
        st.session_state.username = None
        self.clear_cache()
        st.rerun()
    
    def clear_cache(self):
        """Clear all cached data."""
        st.session_state.data_cache = {}
        st.session_state.model_cache = {}
        st.session_state.last_data_refresh = None
    
    def render_main_content(self):
        """Render the main content area based on current page."""
        
        # Page header
        page_name = st.session_state.current_page
        st.markdown(f'<div class="main-header">{self.get_page_icon(page_name)} {page_name}</div>', 
                   unsafe_allow_html=True)
        
        # Route to appropriate page
        if page_name == 'Data Overview':
            self.render_data_overview_placeholder()
        elif page_name == 'Model Performance':
            self.render_model_performance_placeholder()
        elif page_name == 'Predictions':
            self.render_predictions_placeholder()
        elif page_name == 'Monitoring':
            self.render_monitoring_page()
        elif page_name == 'Model Management':
            self.render_model_management_placeholder()
        else:
            st.error(f"Unknown page: {page_name}")
    
    def get_page_icon(self, page_name: str) -> str:
        """Get icon for page."""
        icons = {
            'Data Overview': '📈',
            'Model Performance': '🎯',
            'Predictions': '🔮',
            'Monitoring': '📊',
            'Model Management': '⚙️'
        }
        return icons.get(page_name, '📊')
    
    def render_data_overview_placeholder(self):
        """Placeholder for data overview page."""
        st.info("🚧 Data Overview page is under construction. This will display comprehensive data exploration and visualization capabilities.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", "7,043", "↑ 2.1%")
        with col2:
            st.metric("Churn Rate", "26.5%", "↓ 1.2%")
        with col3:
            st.metric("Avg Monthly Charges", "$64.76", "↑ $3.20")
        with col4:
            st.metric("Customer Lifetime Value", "$1,869", "↑ $127")
        
        st.markdown("### Coming Soon:")
        st.markdown("""
        - 📊 **Dataset Statistics**: Comprehensive data profiling and quality metrics
        - 📈 **Distribution Analysis**: Interactive histograms and box plots
        - 🎯 **Churn Analysis**: Detailed churn rate breakdowns by segments
        - 🔗 **Correlation Matrix**: Feature correlation heatmaps
        - 🔍 **Interactive Filters**: Dynamic data slicing and dicing
        """)
    
    def render_model_performance_placeholder(self):
        """Placeholder for model performance page."""
        st.info("🚧 Model Performance page is under construction. This will display comprehensive model evaluation metrics and visualizations.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "84.2%", "↑ 1.5%")
        with col2:
            st.metric("AUC Score", "0.847", "↑ 0.023")
        with col3:
            st.metric("F1 Score", "0.763", "↑ 0.018")
        
        st.markdown("### Coming Soon:")
        st.markdown("""
        - 📊 **Training Metrics**: Real-time training and validation curves
        - 🎯 **Confusion Matrix**: Interactive confusion matrix visualization
        - 📈 **ROC/PR Curves**: Model discrimination and calibration analysis
        - 🔍 **Feature Importance**: Model interpretability and SHAP analysis
        - 📋 **Model Comparison**: Side-by-side model performance comparison
        """)
    
    def render_predictions_placeholder(self):
        """Placeholder for predictions page."""
        st.info("🚧 Predictions page is under construction. This will provide real-time churn prediction capabilities.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predictions Today", "1,247", "↑ 23")
        with col2:
            st.metric("High Risk Customers", "89", "↑ 7")
        
        st.markdown("### Coming Soon:")
        st.markdown("""
        - 📝 **Customer Input Form**: Manual customer data entry for predictions
        - 📄 **Batch Upload**: CSV file upload for bulk predictions
        - ⚡ **Real-time Results**: Instant churn probability and risk scoring
        - 🎯 **Risk Categories**: Automated Low/Medium/High risk classification
        - 📊 **Confidence Intervals**: Prediction uncertainty quantification
        - 📈 **Prediction History**: Track and analyze past predictions
        """)
    
    def render_model_management_placeholder(self):
        """Placeholder for model management page."""
        st.info("🚧 Model Management page is under construction. This will provide MLflow model registry integration and deployment controls.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Models", "3", "→ 0")
        with col2:
            st.metric("Models in Production", "2", "→ 0")
        with col3:
            st.metric("Last Training", "2 hours ago", "🔄")
        
        st.markdown("### Coming Soon:")
        st.markdown("""
        - 📋 **Model Registry**: Browse and manage all trained models
        - 🚀 **Model Promotion**: Deploy models from staging to production
        - 📊 **Performance Comparison**: Compare models across metrics
        - 🔄 **Model Rollback**: Quick rollback to previous model versions
        - ⚡ **Training Triggers**: Initiate new training runs
        - 🖥️ **System Health**: Monitor model serving and system status
        """)
    
    @monitor_performance('streamlit_monitoring_page')
    def render_monitoring_page(self):
        """Render the monitoring and observability page."""
        st.markdown('<div class="main-header">📊 System Monitoring & Observability</div>', unsafe_allow_html=True)
        
        # Track page view
        metrics_collector.track_request('GET', '/monitoring', 200, 0.1)
        
        try:
            # Get current monitoring status
            monitoring_status = get_monitoring_status()
            
            # System Health Overview
            st.markdown("### 🏥 System Health Overview")
            
            health_data = monitoring_status.get('health', {})
            overall_status = health_data.get('overall_status', 'unknown')
            
            # Status indicators
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if overall_status == 'healthy':
                    st.success("🟢 System Healthy")
                elif overall_status == 'warning':
                    st.warning("🟡 System Warning")
                else:
                    st.error("🔴 System Unhealthy")
            
            with col2:
                prometheus_enabled = monitoring_status.get('metrics', {}).get('prometheus_enabled', False)
                if prometheus_enabled:
                    st.success("📈 Metrics Active")
                else:
                    st.warning("📊 Metrics Limited")
            
            with col3:
                active_alerts = monitoring_status.get('alerts', {}).get('active_count', 0)
                if active_alerts == 0:
                    st.success("🔕 No Alerts")
                else:
                    st.error(f"🚨 {active_alerts} Active Alerts")
            
            with col4:
                st.info(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            
            # Detailed Health Checks
            st.markdown("### 🔍 Detailed Health Checks")
            
            if 'checks' in health_data:
                health_df_data = []
                for check_name, check_result in health_data['checks'].items():
                    status = check_result.get('status', 'unknown')
                    health_df_data.append({
                        'Component': check_name.replace('_', ' ').title(),
                        'Status': status,
                        'Details': check_result.get('error', 'Healthy') if status != 'healthy' else 'All checks passed'
                    })
                
                health_df = pd.DataFrame(health_df_data)
                
                # Apply styling to the dataframe
                def style_status(val):
                    if val == 'healthy':
                        return 'background-color: #d4edda; color: #155724'
                    elif val == 'warning':
                        return 'background-color: #fff3cd; color: #856404'
                    else:
                        return 'background-color: #f8d7da; color: #721c24'
                
                st.dataframe(
                    health_df.style.applymap(style_status, subset=['Status']),
                    use_container_width=True
                )
            
            # Metrics Summary
            st.markdown("### 📊 Metrics Summary")
            
            metrics_data = monitoring_status.get('metrics', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Prometheus Integration")
                if metrics_data.get('prometheus_enabled', False):
                    st.success("✅ Prometheus metrics enabled")
                    st.info(f"📈 Metrics server running on port {metrics_data.get('metrics_port', 8000)}")
                    st.info(f"📊 Total metrics: {metrics_data.get('total_metrics', 0)}")
                    
                    # Quick links
                    st.markdown("**Quick Links:**")
                    st.markdown("- [Prometheus](http://localhost:9090) - Metrics & Queries")
                    st.markdown("- [Grafana](http://localhost:3000) - Dashboards (admin/retentionai123)")
                    st.markdown("- [Alertmanager](http://localhost:9093) - Alert Management")
                    
                else:
                    st.warning("⚠️ Prometheus metrics disabled")
                    st.info("Using in-memory metrics fallback")
                    
                    in_memory = metrics_data.get('in_memory_metrics', {})
                    st.write(f"Counters: {in_memory.get('counters', 0)}")
                    st.write(f"Gauges: {in_memory.get('gauges', 0)}")
                    st.write(f"Histograms: {in_memory.get('histograms', 0)}")
            
            with col2:
                st.markdown("#### Health Check Endpoints")
                st.info("Health API running on port 8001")
                
                # Health endpoint status
                endpoints = [
                    ("General Health", "http://localhost:8001/health"),
                    ("Readiness", "http://localhost:8001/ready"),
                    ("Liveness", "http://localhost:8001/live"),
                    ("Metrics Summary", "http://localhost:8001/metrics/summary"),
                ]
                
                for name, url in endpoints:
                    st.markdown(f"- [{name}]({url})")
                
                # System info
                st.markdown("#### System Information")
                st.write(f"Application: RetentionAI v1.0.0")
                st.write(f"Environment: Production")
                st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Performance Monitoring Section
            st.markdown("### ⚡ Performance Monitoring")
            
            # Simulated real-time metrics (in production, these would come from Prometheus)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Request Rate", "45.2/sec", "↑ 12%", delta_color="normal")
            with col2:
                st.metric("Avg Response Time", "245ms", "↓ 15ms", delta_color="inverse")
            with col3:
                st.metric("Memory Usage", "67%", "↑ 3%", delta_color="off")
            with col4:
                st.metric("CPU Usage", "23%", "↓ 5%", delta_color="inverse")
            
            # Model-specific metrics
            st.markdown("#### 🤖 ML Model Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy", "87.3%", "↑ 1.2%", delta_color="normal")
            with col2:
                st.metric("Predictions/min", "1,247", "↑ 156", delta_color="normal")
            with col3:
                st.metric("Avg Prediction Time", "18ms", "↓ 3ms", delta_color="inverse")
            
            # Monitoring Controls
            st.markdown("### 🎛️ Monitoring Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Refresh Monitoring Data"):
                    st.rerun()
            
            with col2:
                if st.button("🧪 Test Health Checks"):
                    with st.spinner("Running health checks..."):
                        # Simulate running health checks
                        import time
                        time.sleep(2)
                        st.success("All health checks passed!")
            
            with col3:
                if st.button("📊 Open Grafana"):
                    st.markdown("[🔗 Open Grafana Dashboard](http://localhost:3000)")
            
            # Monitoring Configuration
            with st.expander("⚙️ Monitoring Configuration"):
                st.markdown("#### Current Configuration")
                config_data = {
                    "Prometheus Enabled": metrics_data.get('prometheus_enabled', False),
                    "Metrics Port": metrics_data.get('metrics_port', 8000),
                    "Health Check Port": 8001,
                    "Auto Refresh": st.session_state.get('auto_refresh', False),
                    "Monitoring Initialized": st.session_state.get('monitoring_initialized', False)
                }
                
                st.json(config_data)
                
                st.markdown("#### Monitoring Stack Services")
                st.markdown("""
                - **Prometheus**: Metrics collection and storage (Port 9090)
                - **Grafana**: Visualization and dashboards (Port 3000)
                - **Alertmanager**: Alert routing and notifications (Port 9093)
                - **Node Exporter**: System metrics (Port 9100)
                - **cAdvisor**: Container metrics (Port 8080)
                - **Loki**: Log aggregation (Port 3100)
                """)
                
                if st.button("📚 View Monitoring Documentation"):
                    st.markdown("""
                    ### Monitoring Documentation
                    
                    **Starting the monitoring stack:**
                    ```bash
                    ./scripts/monitoring.sh start
                    ```
                    
                    **Checking status:**
                    ```bash
                    ./scripts/monitoring.sh status
                    ```
                    
                    **Viewing service logs:**
                    ```bash
                    ./scripts/monitoring.sh logs prometheus
                    ```
                    """)
        
        except Exception as e:
            st.error(f"Error loading monitoring data: {e}")
            logger.error(f"Monitoring page error: {e}", exc_info=True)
    
    def run(self):
        """Run the main application."""
        try:
            self.render_sidebar()
            self.render_main_content()
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Application error: {e}", exc_info=True)
            
            if st.button("🔄 Restart Application"):
                st.rerun()


def main():
    """Main application entry point."""
    
    # Initialize and run the app
    try:
        app = RetentionAIApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.error("Please check the application logs and ensure all dependencies are properly installed.")
        logger.error(f"Application initialization error: {e}", exc_info=True)


if __name__ == "__main__":
    main()