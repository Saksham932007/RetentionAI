# Getting Started with RetentionAI

## Welcome to RetentionAI! 🎉

This guide will help you get RetentionAI up and running in just a few minutes. Whether you're a data scientist, developer, or business user, you'll be making churn predictions in no time.

## Quick Start Options

### 🚀 Option 1: Docker (Recommended)

The fastest way to get started with RetentionAI:

```bash
# Clone the repository
git clone https://github.com/Saksham932007/RetentionAI.git
cd RetentionAI

# Start the application with Docker
./scripts/deploy.sh dev

# Wait for services to start (about 2-3 minutes)
# Access the application at http://localhost:8501
```

### 🐍 Option 2: Local Python Setup

For development and customization:

```bash
# Clone the repository
git clone https://github.com/Saksham932007/RetentionAI.git
cd RetentionAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up the database
python src/database.py

# Run the application
streamlit run src/app.py
```

### ☸️ Option 3: Kubernetes (Production)

For production deployments:

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
kubectl port-forward service/retentionai-service 8501:8501

# Access at http://localhost:8501
```

## First Time Setup

### 1. Verify Installation

After starting RetentionAI, open your browser and navigate to:
- **Local**: http://localhost:8501
- **Docker**: http://localhost:8501
- **Production**: https://your-domain.com

You should see the RetentionAI dashboard with a welcome message and system status indicators.

### 2. Check System Health

The dashboard will display system health indicators:
- 🟢 **Green**: All systems operational
- 🟡 **Yellow**: Minor issues detected
- 🔴 **Red**: Critical issues requiring attention

### 3. Load Sample Data

RetentionAI comes with sample data for testing:

1. Navigate to the **Prediction** section
2. Try making a single customer prediction
3. Upload the sample CSV file from `data/sample_customers.csv`

## Your First Prediction

### Single Customer Prediction

Let's make your first churn prediction:

1. **Go to the Prediction Center**
   - Click on "Single Prediction" in the sidebar

2. **Enter Customer Information**
   ```
   Gender: Female
   Senior Citizen: No
   Partner: No
   Dependents: No
   Tenure: 6 months
   Phone Service: Yes
   Multiple Lines: No
   Internet Service: DSL
   Online Security: No
   Online Backup: No
   Device Protection: No
   Tech Support: No
   Streaming TV: No
   Streaming Movies: No
   Contract: Month-to-month
   Paperless Billing: Yes
   Payment Method: Electronic check
   Monthly Charges: $45.00
   Total Charges: $270.00
   ```

3. **Click "Predict Churn"**

4. **View Results**
   You'll see:
   - **Churn Probability**: e.g., 73% (High Risk)
   - **Feature Importance**: Which factors matter most
   - **SHAP Explanations**: How each feature influences the prediction
   - **Recommendations**: Suggested retention actions

### Batch Prediction

Try processing multiple customers:

1. **Go to Batch Prediction**
2. **Download the CSV template** or use the sample file
3. **Upload your file**
4. **Review results** and download the output

## Understanding the Interface

### 🏠 Dashboard Overview

The main dashboard shows:
- **System Status**: Health indicators and uptime
- **Key Metrics**: Total customers, high-risk count, model accuracy
- **Recent Predictions**: Latest prediction results
- **Quick Actions**: Common tasks and shortcuts

### 🔮 Prediction Center

Your prediction workspace with:
- **Single Prediction**: Individual customer analysis
- **Batch Processing**: Bulk prediction uploads
- **History**: Previous prediction results
- **Templates**: CSV templates and examples

### 📊 Analytics Hub

Business intelligence features:
- **Risk Distribution**: Customer risk breakdown
- **Feature Insights**: Most important churn factors
- **Performance Metrics**: Model accuracy and trends
- **Business Impact**: ROI and retention analytics

### ⚙️ Settings

Configuration options:
- **Thresholds**: Adjust risk level boundaries
- **Notifications**: Alert preferences
- **Data Sources**: Integration settings
- **User Management**: Access control (admin only)

## Key Concepts

### Churn Prediction

**What is customer churn?**
Customer churn occurs when customers stop using your service. RetentionAI predicts which customers are likely to churn so you can take proactive retention actions.

**How does it work?**
RetentionAI uses advanced machine learning (XGBoost ensemble) to analyze customer characteristics and behavior patterns to predict churn probability.

### Risk Levels

RetentionAI categorizes customers into three risk levels:

- **🔴 High Risk (>70%)**: Immediate attention required
  - Schedule retention call within 24 hours
  - Offer incentives or contract improvements
  - Monitor closely for service issues

- **🟡 Medium Risk (30-70%)**: Monitor and engage
  - Send targeted promotions
  - Improve service experience
  - Consider upselling opportunities

- **🟢 Low Risk (<30%)**: Maintain satisfaction
  - Continue standard service
  - Look for expansion opportunities
  - Gather feedback for improvements

### Feature Importance

The model considers various customer attributes:

1. **Contract Type** (Most Important)
   - Month-to-month contracts have higher churn risk
   - Longer contracts indicate commitment

2. **Tenure** (Very Important)
   - Newer customers are more likely to churn
   - Loyalty increases with time

3. **Payment Method** (Important)
   - Electronic check has higher risk
   - Automatic payments indicate stability

4. **Charges** (Moderately Important)
   - Higher monthly charges can increase risk
   - Value perception matters

5. **Services** (Moderately Important)
   - More services indicate deeper engagement
   - Premium services may increase satisfaction

### SHAP Explanations

SHAP (SHapley Additive exPlanations) values show how each feature influences the prediction:

- **Positive values (red bars)**: Increase churn probability
- **Negative values (blue bars)**: Decrease churn probability
- **Magnitude**: Shows the strength of influence

Example interpretation:
```
Contract=Month-to-month: +0.45 → Strong positive influence (increases churn risk)
Tenure=6 months: +0.23 → Moderate positive influence (new customer risk)
Payment=Electronic check: +0.15 → Weak positive influence
Total Charges=$270: -0.12 → Weak negative influence (some investment)
```

## Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Application settings
APP_NAME=RetentionAI
APP_VERSION=1.0.0
DEBUG=false

# Database settings
DATABASE_URL=sqlite:///data/retentionai.db

# Model settings
MODEL_PATH=models/best_model.pkl
FEATURE_SCALER_PATH=models/scaler.pkl

# Monitoring settings
PROMETHEUS_PORT=8000
GRAFANA_PORT=3000

# Security settings
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Model Configuration

Adjust prediction thresholds in `config/model_config.yaml`:

```yaml
model:
  name: "XGBoost Ensemble"
  version: "2.1.0"
  
thresholds:
  high_risk: 0.70
  low_risk: 0.30
  
features:
  required: [
    "tenure", "monthly_charges", "contract",
    "payment_method", "internet_service"
  ]
  
preprocessing:
  scaling: "standard"
  encoding: "target"
```

### Database Configuration

For production databases, update `config/database.yaml`:

```yaml
database:
  type: "postgresql"  # or "mysql", "sqlite"
  host: "localhost"
  port: 5432
  database: "retentionai"
  username: "retentionai_user"
  password: "secure_password"
  
connection_pool:
  min_size: 2
  max_size: 20
  timeout: 30
```

## Integration

### CRM Integration

Connect RetentionAI to your CRM system:

1. **API Integration**
   ```python
   # Example CRM connector
   from retentionai.integrations import CRMConnector
   
   connector = CRMConnector(
       api_key="your-crm-api-key",
       base_url="https://your-crm.com/api"
   )
   
   # Sync customer data
   connector.sync_customers()
   
   # Send retention alerts
   connector.send_alert(customer_id, risk_score)
   ```

2. **Webhook Setup**
   Configure webhooks to receive real-time updates:
   ```bash
   curl -X POST https://your-retentionai.com/webhooks/crm \
     -H "Content-Type: application/json" \
     -d '{"customer_id": "CUST_001", "event": "profile_update"}'
   ```

### BI Tool Integration

Connect to business intelligence tools:

1. **Database Connection**
   - Connect Tableau, Power BI, or Looker to the RetentionAI database
   - Use read-only credentials for security
   - Query the `predictions` and `customers` tables

2. **API Integration**
   - Use the REST API for real-time data
   - Implement custom dashboards
   - Automate report generation

### Email/SMS Integration

Set up automated communications:

1. **Email Alerts**
   ```python
   # Configure SMTP settings
   EMAIL_HOST = "smtp.gmail.com"
   EMAIL_PORT = 587
   EMAIL_USE_TLS = True
   EMAIL_HOST_USER = "alerts@yourcompany.com"
   EMAIL_HOST_PASSWORD = "app-specific-password"
   ```

2. **SMS Notifications**
   ```python
   # Twilio integration example
   from twilio.rest import Client
   
   client = Client(account_sid, auth_token)
   message = client.messages.create(
       body="High-risk customer alert: CUST_001",
       from_="+1234567890",
       to="+0987654321"
   )
   ```

## Troubleshooting

### Common Issues

#### Application Won't Start

**Problem**: Error messages during startup
**Solutions**:
1. Check Python version (requires 3.10+)
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure database file permissions: `chmod 664 data/retentionai.db`
4. Check port availability: `lsof -i :8501`

#### Prediction Errors

**Problem**: "Prediction failed" error messages
**Solutions**:
1. Verify model file exists: `ls -la models/best_model.pkl`
2. Check data format matches requirements
3. Ensure all required features are present
4. Review application logs: `tail -f logs/app.log`

#### Performance Issues

**Problem**: Slow response times, high memory usage
**Solutions**:
1. Restart the application to clear memory
2. Reduce batch size for large uploads
3. Check system resources: `htop`, `free -h`
4. Monitor database performance

#### Docker Issues

**Problem**: Container startup failures
**Solutions**:
1. Check Docker daemon is running: `docker info`
2. Verify image build: `docker-compose build --no-cache`
3. Check port conflicts: `docker-compose ps`
4. Review container logs: `docker-compose logs app`

### Getting Help

#### Self-Service Resources
- **Documentation**: Browse the `/docs` folder
- **FAQ**: Check common questions in `docs/faq.md`
- **Video Tutorials**: Available in the help section
- **Sample Data**: Use provided examples in `/data/samples`

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussion Forum**: Community Q&A and best practices
- **Stack Overflow**: Use the `retentionai` tag

#### Enterprise Support
- **Email**: support@retentionai.com
- **Priority Support**: Available for enterprise customers
- **Custom Training**: On-site training and consulting
- **Professional Services**: Custom implementation and integration

## Next Steps

### Explore Advanced Features

1. **Model Customization**
   - Train models with your own data
   - Adjust hyperparameters
   - Implement custom features

2. **Advanced Analytics**
   - Set up custom dashboards
   - Create automated reports
   - Build prediction pipelines

3. **Production Deployment**
   - Deploy to cloud platforms
   - Set up monitoring and alerting
   - Implement CI/CD pipelines

### Learn More

1. **User Manual**: Detailed feature documentation
2. **API Reference**: Complete API documentation
3. **Architecture Guide**: System design and components
4. **Best Practices**: Operational recommendations

### Join the Community

1. **GitHub**: Star the repository and contribute
2. **LinkedIn**: Follow for updates and insights
3. **Newsletter**: Subscribe for tips and new features
4. **Slack**: Join the community workspace

## Success Checklist

Mark off each item as you complete it:

- [ ] Successfully installed RetentionAI
- [ ] Accessed the web interface
- [ ] Made your first single prediction
- [ ] Uploaded and processed a batch file
- [ ] Reviewed SHAP explanations
- [ ] Explored the analytics dashboard
- [ ] Configured basic settings
- [ ] Set up monitoring (optional)
- [ ] Integrated with existing systems (optional)
- [ ] Trained team members on usage

**Congratulations! 🎉** You're now ready to use RetentionAI to predict customer churn and improve retention rates.

---

*Need help? Contact us at support@retentionai.com or visit our documentation at docs.retentionai.com*