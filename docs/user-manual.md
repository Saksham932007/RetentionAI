# User Manual - RetentionAI

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview](#dashboard-overview)
3. [Making Predictions](#making-predictions)
4. [Understanding Results](#understanding-results)
5. [Data Upload](#data-upload)
6. [Analytics & Reports](#analytics--reports)
7. [Settings & Configuration](#settings--configuration)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Accessing RetentionAI

1. **Open your web browser** and navigate to RetentionAI
   - Development: `http://localhost:8501`
   - Production: `https://retentionai.com`

2. **Login** (if authentication is enabled)
   - Enter your username and password
   - Click "Sign In"

3. **Welcome Screen** will display system status and quick links

### System Requirements

- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Internet Connection**: Required for real-time features
- **Screen Resolution**: Minimum 1024x768 (recommended 1920x1080)

## Dashboard Overview

### Main Navigation

The RetentionAI interface consists of several main sections:

#### 🏠 Home Dashboard
- **System Status**: Health indicators and uptime
- **Quick Stats**: Total customers, high-risk count, model accuracy
- **Recent Activity**: Latest predictions and alerts
- **Performance Metrics**: Real-time system performance

#### 🔮 Prediction Center
- **Single Prediction**: Predict individual customer churn
- **Batch Prediction**: Upload CSV for bulk predictions
- **Real-time Scoring**: API-based prediction interface

#### 📊 Analytics Hub
- **Risk Distribution**: Visual breakdown of customer risk levels
- **Feature Insights**: Most important churn indicators
- **Trend Analysis**: Historical churn patterns
- **Business Impact**: ROI and retention metrics

#### 🔍 Customer Explorer
- **Search Customers**: Find specific customer profiles
- **Risk Segmentation**: Browse customers by risk level
- **Customer Journey**: View individual customer history

#### ⚙️ Settings Panel
- **Model Configuration**: Adjust prediction thresholds
- **Data Sources**: Manage data connections
- **Notifications**: Configure alerts and reports
- **User Management**: Access control (admin only)

### Status Indicators

RetentionAI uses color-coded status indicators throughout the interface:

- 🟢 **Green**: Healthy, optimal, low risk
- 🟡 **Yellow**: Warning, attention needed, medium risk
- 🔴 **Red**: Critical, action required, high risk
- 🔵 **Blue**: Informational, neutral status
- ⚫ **Gray**: Inactive, disabled, unknown

## Making Predictions

### Single Customer Prediction

1. **Navigate** to the Prediction Center
2. **Select** "Single Customer Prediction"
3. **Fill in Customer Data**:

#### Personal Information
- **Gender**: Male/Female
- **Senior Citizen**: Yes/No (65+ years old)
- **Partner**: Yes/No (has a partner/spouse)
- **Dependents**: Yes/No (has dependents/children)

#### Service Information
- **Tenure**: Number of months as customer (0-72)
- **Phone Service**: Yes/No
- **Multiple Lines**: Yes/No/No phone service
- **Internet Service**: DSL/Fiber optic/No

#### Additional Services
- **Online Security**: Yes/No/No internet service
- **Online Backup**: Yes/No/No internet service
- **Device Protection**: Yes/No/No internet service
- **Tech Support**: Yes/No/No internet service
- **Streaming TV**: Yes/No/No internet service
- **Streaming Movies**: Yes/No/No internet service

#### Billing Information
- **Contract**: Month-to-month/One year/Two year
- **Paperless Billing**: Yes/No
- **Payment Method**: Electronic check/Mailed check/Bank transfer/Credit card
- **Monthly Charges**: Dollar amount (0-120)
- **Total Charges**: Lifetime value (0-9000)

4. **Click** "Predict Churn" button
5. **View Results** (see Understanding Results section)

### Example Customer Profile

Here's an example of a high-risk customer profile:

```
Gender: Female
Senior Citizen: No
Partner: No
Dependents: No
Tenure: 3 months
Phone Service: Yes
Multiple Lines: No
Internet Service: Fiber optic
Online Security: No
Online Backup: No
Device Protection: No
Tech Support: No
Streaming TV: Yes
Streaming Movies: Yes
Contract: Month-to-month
Paperless Billing: Yes
Payment Method: Electronic check
Monthly Charges: $85.00
Total Charges: $255.00
```

This profile typically results in high churn probability due to:
- Short tenure (new customer)
- Month-to-month contract
- Electronic check payment
- No additional security services
- High monthly charges for limited tenure

### Batch Prediction

1. **Prepare CSV File** with required columns:
   ```
   customer_id,gender,senior_citizen,partner,dependents,tenure,
   phone_service,multiple_lines,internet_service,online_security,
   online_backup,device_protection,tech_support,streaming_tv,
   streaming_movies,contract,paperless_billing,payment_method,
   monthly_charges,total_charges
   ```

2. **Navigate** to "Batch Prediction" section
3. **Upload File** using drag-and-drop or file browser
4. **Review** data preview and validation results
5. **Click** "Process Batch" to run predictions
6. **Download Results** as CSV or view in interface

### CSV Template

Download the CSV template to ensure proper formatting:

| Column | Type | Values | Example |
|--------|------|--------|---------|
| customer_id | String | Unique identifier | "CUST_001" |
| gender | String | "Male" or "Female" | "Female" |
| senior_citizen | Integer | 0 or 1 | 0 |
| partner | String | "Yes" or "No" | "No" |
| dependents | String | "Yes" or "No" | "No" |
| tenure | Integer | 0-72 | 12 |
| phone_service | String | "Yes" or "No" | "Yes" |
| multiple_lines | String | "Yes", "No", "No phone service" | "No" |
| internet_service | String | "DSL", "Fiber optic", "No" | "DSL" |
| online_security | String | "Yes", "No", "No internet service" | "No" |
| online_backup | String | "Yes", "No", "No internet service" | "Yes" |
| device_protection | String | "Yes", "No", "No internet service" | "No" |
| tech_support | String | "Yes", "No", "No internet service" | "No" |
| streaming_tv | String | "Yes", "No", "No internet service" | "No" |
| streaming_movies | String | "Yes", "No", "No internet service" | "No" |
| contract | String | "Month-to-month", "One year", "Two year" | "Month-to-month" |
| paperless_billing | String | "Yes" or "No" | "Yes" |
| payment_method | String | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)" | "Electronic check" |
| monthly_charges | Float | 0.00-120.00 | 29.85 |
| total_charges | Float | 0.00-9000.00 | 358.20 |

## Understanding Results

### Prediction Output

When you make a prediction, RetentionAI provides comprehensive results:

#### Primary Metrics
- **Churn Probability**: 0.00-1.00 (0% to 100% likelihood)
- **Prediction**: "Yes" (will churn) or "No" (will not churn)
- **Confidence Level**: High/Medium/Low
- **Risk Category**: High/Medium/Low

#### Risk Thresholds
- **High Risk**: Probability > 0.70 (70%)
- **Medium Risk**: Probability 0.30-0.70 (30-70%)
- **Low Risk**: Probability < 0.30 (30%)

#### Feature Importance
Shows which factors most influenced the prediction:

```
1. Contract Type: 25% influence
2. Tenure: 18% influence  
3. Monthly Charges: 15% influence
4. Payment Method: 12% influence
5. Internet Service: 10% influence
```

#### SHAP Values
Explains how each feature pushed the prediction toward or away from churn:

- **Positive values** (red): Increase churn probability
- **Negative values** (blue): Decrease churn probability
- **Magnitude**: Shows strength of influence

Example SHAP interpretation:
- Contract=Month-to-month: +0.45 (strong positive influence)
- Tenure=12 months: -0.23 (moderate negative influence)
- Payment=Electronic check: +0.15 (weak positive influence)

#### Recommendations

Based on the prediction, RetentionAI suggests actionable interventions:

**High Risk Customer:**
- "Offer long-term contract incentives"
- "Provide payment method alternatives"
- "Schedule retention call within 24 hours"

**Medium Risk Customer:**
- "Monitor usage patterns closely"
- "Offer service upgrades or bundles"
- "Send targeted promotional offers"

**Low Risk Customer:**
- "Continue standard service"
- "Consider upselling opportunities"
- "Maintain satisfaction surveys"

### Interpreting Confidence Levels

#### High Confidence (>90%)
- Model is very certain about the prediction
- Strong patterns in customer data
- Recommendations should be prioritized

#### Medium Confidence (70-90%)
- Model has good certainty
- Some conflicting signals in data
- Monitor customer closely

#### Low Confidence (<70%)
- Model is uncertain
- Limited or conflicting data patterns
- Consider additional data collection

## Data Upload

### Supported File Formats

- **CSV**: Comma-separated values (recommended)
- **Excel**: .xlsx files (limited to 10,000 rows)
- **TSV**: Tab-separated values

### File Size Limits

- **Single File**: Maximum 50MB
- **Batch Processing**: Up to 100,000 customers
- **Recommended**: 1,000-10,000 customers for optimal performance

### Data Validation

RetentionAI automatically validates uploaded data:

#### Required Fields Check
- Ensures all required columns are present
- Identifies missing critical data

#### Data Type Validation
- Verifies numeric fields contain numbers
- Checks categorical fields match expected values

#### Range Validation
- Tenure: 0-72 months
- Monthly Charges: $0-$120
- Total Charges: $0-$9,000

#### Quality Checks
- Identifies outliers and unusual values
- Flags potentially incorrect data
- Suggests data corrections

### Common Upload Issues

#### Missing Columns
**Error**: "Required column 'tenure' not found"
**Solution**: Ensure CSV headers exactly match template

#### Invalid Values
**Error**: "Invalid contract value 'Monthly'"
**Solution**: Use exact values: "Month-to-month", "One year", "Two year"

#### Data Type Errors
**Error**: "Monthly charges must be numeric"
**Solution**: Remove currency symbols, use decimal notation

#### Encoding Issues
**Error**: "File encoding not supported"
**Solution**: Save CSV with UTF-8 encoding

## Analytics & Reports

### Risk Distribution Dashboard

View the breakdown of your customer base by risk level:

#### High-Risk Customers
- **Count**: Number of customers with >70% churn probability
- **Percentage**: Proportion of total customer base
- **Trend**: Change from previous period
- **Revenue Impact**: Potential revenue at risk

#### Medium-Risk Customers  
- **Count**: Customers with 30-70% churn probability
- **Watch List**: Customers trending toward high risk
- **Opportunity**: Candidates for retention programs

#### Low-Risk Customers
- **Count**: Customers with <30% churn probability
- **Stability**: Loyal customer base
- **Upselling**: Opportunities for service expansion

### Feature Impact Analysis

Understand which factors drive churn in your customer base:

#### Global Feature Importance
Ranked list of most influential factors:
1. Contract type (month-to-month highest risk)
2. Tenure (newer customers higher risk)
3. Payment method (electronic check highest risk)
4. Monthly charges (higher charges increase risk)
5. Total charges (relationship with tenure)

#### Segment Analysis
Different customer segments may have different churn drivers:

**New Customers (0-12 months)**
- Contract type most important
- Payment method secondary factor

**Established Customers (13+ months)**  
- Service satisfaction becomes key
- Price sensitivity increases

### Business Impact Metrics

#### Revenue at Risk
- **Total Monthly Revenue**: Sum of monthly charges for all customers
- **High-Risk Revenue**: Monthly charges from high-risk customers
- **Potential Loss**: Estimated revenue loss if high-risk customers churn

#### Retention ROI
- **Intervention Cost**: Cost per retention attempt
- **Success Rate**: Percentage of successful retentions
- **Customer Lifetime Value**: Average value of retained customer
- **Net Benefit**: (Retained Value × Success Rate) - Intervention Cost

### Trend Analysis

#### Historical Patterns
- Monthly churn rates over time
- Seasonal variations in churn
- Impact of business changes on retention

#### Predictive Trends
- Forecasted churn rates
- Early warning indicators
- Capacity planning for retention efforts

## Settings & Configuration

### Model Configuration

#### Prediction Thresholds
Adjust the boundaries for risk categorization:
- **High Risk Threshold**: Default 0.70 (range 0.60-0.85)
- **Low Risk Threshold**: Default 0.30 (range 0.15-0.45)

#### Model Selection
Choose which model to use for predictions:
- **Production Model**: Current live model
- **Challenger Model**: Alternative model for A/B testing
- **Custom Model**: User-uploaded model

### Data Sources

#### Database Connections
Configure connections to customer databases:
- **Connection String**: Database URL and credentials
- **Refresh Schedule**: Automatic data update frequency
- **Data Mapping**: Field mapping configuration

#### API Integrations
Set up real-time data feeds:
- **CRM Integration**: Salesforce, HubSpot connections
- **Billing System**: Payment and subscription data
- **Support System**: Customer service interactions

### Notification Settings

#### Alert Thresholds
Configure when to receive notifications:
- **High-Risk Customer**: Individual customer exceeds threshold
- **Batch Risk Increase**: Significant increase in at-risk customers
- **Model Performance**: Accuracy drops below threshold
- **System Health**: Application or infrastructure issues

#### Notification Channels
Choose how to receive alerts:
- **Email**: Send to specified email addresses
- **Slack**: Post to designated Slack channels
- **Webhook**: Send HTTP POST to external systems
- **In-App**: Display in RetentionAI interface

### User Management (Admin Only)

#### User Roles
- **Admin**: Full system access and configuration
- **Analyst**: Analytics, reporting, and predictions
- **Viewer**: Read-only access to dashboards
- **API User**: Programmatic access only

#### Access Control
- **Feature Access**: Control which sections users can access
- **Data Access**: Restrict sensitive customer information
- **Export Permissions**: Control data download capabilities

## Troubleshooting

### Common Issues

#### Slow Performance
**Symptoms**: Predictions take >30 seconds, dashboard loads slowly
**Causes**: Large batch uploads, high system load, network issues
**Solutions**:
- Reduce batch size to <5,000 customers
- Check internet connection
- Try again during off-peak hours
- Contact support if persistent

#### Prediction Errors
**Symptoms**: "Prediction failed" error message
**Causes**: Invalid data format, missing required fields, system overload
**Solutions**:
- Verify all required fields are present
- Check data format matches template
- Retry with smaller batch
- Review error logs for details

#### Login Issues
**Symptoms**: Cannot access application, authentication failures
**Causes**: Expired session, incorrect credentials, server issues
**Solutions**:
- Clear browser cache and cookies
- Verify username and password
- Try incognito/private browsing mode
- Contact administrator for password reset

#### File Upload Problems
**Symptoms**: "File upload failed" or "Invalid file format"
**Causes**: File too large, unsupported format, network timeout
**Solutions**:
- Ensure file is <50MB
- Use CSV format when possible
- Check file encoding (use UTF-8)
- Verify internet connection stability

### Browser Compatibility

#### Recommended Browsers
- **Chrome 90+**: Full feature support
- **Firefox 88+**: Full feature support
- **Safari 14+**: Full feature support (Mac only)
- **Edge 90+**: Full feature support

#### Unsupported Browsers
- Internet Explorer (all versions)
- Chrome <85
- Firefox <80

#### Mobile Support
RetentionAI is optimized for desktop use. Mobile browsers have limited functionality:
- **Viewing**: Dashboard and results viewing supported
- **Predictions**: Single predictions supported
- **Uploads**: File uploads not recommended on mobile

### Error Messages

#### Data Validation Errors

**"Invalid contract value"**
- Use exact values: "Month-to-month", "One year", "Two year"
- Check for extra spaces or special characters

**"Tenure must be between 0 and 72"**
- Ensure tenure is in months, not years
- Remove negative values or values >72

**"Missing required column"**
- Download CSV template for correct headers
- Ensure no columns are missing from upload file

#### System Errors

**"Model temporarily unavailable"**
- Model is being updated or retrained
- Try again in 5-10 minutes
- Contact support if error persists >30 minutes

**"Service temporarily overloaded"**  
- High system usage, predictions queued
- Reduce batch size or wait for off-peak hours
- Priority support available for enterprise users

**"Database connection failed"**
- Temporary database maintenance
- Historical data may be temporarily unavailable
- Real-time predictions should continue working

### Getting Help

#### Self-Service Resources
- **FAQ**: Common questions and answers
- **Video Tutorials**: Step-by-step guides
- **API Documentation**: Developer resources
- **Community Forum**: User discussions

#### Support Channels
- **Email**: [support@retentionai.com](mailto:support@retentionai.com)
- **Live Chat**: Available 9 AM - 5 PM EST (business days)
- **Phone**: 1-800-RETENTION (enterprise customers)
- **Emergency**: [emergency@retentionai.com](mailto:emergency@retentionai.com) (critical issues only)

#### Information to Include
When contacting support, please provide:
- **User ID**: Your login username
- **Timestamp**: When the issue occurred
- **Browser**: Browser name and version
- **Error Message**: Exact error text
- **Steps to Reproduce**: What you were trying to do
- **Screenshots**: Visual evidence of the issue

### Best Practices

#### Data Quality
- **Regular Updates**: Refresh customer data weekly
- **Data Validation**: Review data quality reports monthly
- **Field Consistency**: Ensure consistent data entry standards
- **Historical Tracking**: Maintain data lineage and change logs

#### Prediction Usage
- **Threshold Tuning**: Adjust risk thresholds based on business outcomes
- **Validation**: Compare predictions to actual churn outcomes
- **Segmentation**: Use different strategies for different risk levels
- **Automation**: Set up automatic alerts for high-risk customers

#### System Maintenance
- **Regular Backups**: Download important reports and data
- **Browser Updates**: Keep browsers updated for best performance
- **Cache Clearing**: Clear browser cache monthly
- **Password Security**: Use strong passwords and enable 2FA when available

---

*For additional help or feature requests, please contact our support team or visit the RetentionAI community forum.*