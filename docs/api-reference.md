# API Reference - RetentionAI

## Overview

RetentionAI provides a comprehensive REST API for customer churn prediction and management. This reference documents all available endpoints, request/response formats, and usage examples.

## Base URL

```
Production: https://retentionai.com/api/v1
Development: http://localhost:8501/api/v1
```

## Authentication

Currently, the API uses session-based authentication. Future versions will support API keys.

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

## Health Endpoints

### Health Check

Check the overall health of the system.

```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-22T10:30:00Z",
  "version": "1.0.0",
  "uptime": 3600,
  "checks": {
    "database": "healthy",
    "model": "healthy",
    "memory": "healthy",
    "disk": "healthy"
  }
}
```

### Readiness Check

Check if the system is ready to serve requests.

```http
GET /api/v1/ready
```

**Response:**
```json
{
  "status": "ready",
  "components": {
    "model_loaded": true,
    "database_connected": true,
    "feature_store_ready": true
  }
}
```

### Liveness Check

Check if the system is alive and responsive.

```http
GET /api/v1/live
```

**Response:**
```json
{
  "status": "alive",
  "timestamp": "2025-11-22T10:30:00Z"
}
```

## Prediction Endpoints

### Single Customer Prediction

Predict churn probability for a single customer.

```http
POST /api/v1/predict
Content-Type: application/json

{
  "customer_id": "CUST_001",
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "tenure": 12,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "DSL",
  "online_security": "No",
  "online_backup": "Yes",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 29.85,
  "total_charges": 358.2
}
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.73,
  "churn_prediction": "Yes",
  "confidence": "High",
  "risk_level": "High",
  "feature_importance": {
    "contract": 0.25,
    "tenure": 0.18,
    "monthly_charges": 0.15,
    "payment_method": 0.12,
    "internet_service": 0.10
  },
  "shap_values": {
    "contract": 0.45,
    "tenure": -0.23,
    "monthly_charges": 0.18,
    "payment_method": 0.15,
    "internet_service": 0.08
  },
  "recommendations": [
    "Offer long-term contract incentives",
    "Provide payment method alternatives",
    "Consider service bundling options"
  ]
}
```

### Batch Prediction

Predict churn probability for multiple customers.

```http
POST /api/v1/predict/batch
Content-Type: application/json

{
  "customers": [
    {
      "customer_id": "CUST_001",
      "gender": "Female",
      ...
    },
    {
      "customer_id": "CUST_002",
      "gender": "Male",
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST_001",
      "churn_probability": 0.73,
      "churn_prediction": "Yes"
    },
    {
      "customer_id": "CUST_002",
      "churn_probability": 0.23,
      "churn_prediction": "No"
    }
  ],
  "summary": {
    "total_customers": 2,
    "high_risk": 1,
    "medium_risk": 0,
    "low_risk": 1,
    "avg_churn_probability": 0.48
  }
}
```

### Explanation Endpoint

Get detailed SHAP explanations for a prediction.

```http
POST /api/v1/explain
Content-Type: application/json

{
  "customer_id": "CUST_001",
  "features": {
    "tenure": 12,
    "monthly_charges": 29.85,
    ...
  }
}
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "base_value": 0.26,
  "prediction": 0.73,
  "shap_values": {
    "contract": 0.45,
    "tenure": -0.23,
    "monthly_charges": 0.18
  },
  "feature_contributions": [
    {
      "feature": "contract",
      "value": "Month-to-month",
      "contribution": 0.45,
      "direction": "increases_churn"
    }
  ]
}
```

## Customer Management

### Get Customer Profile

Retrieve detailed customer information and history.

```http
GET /api/v1/customers/{customer_id}
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "profile": {
    "gender": "Female",
    "age_group": "Adult",
    "tenure_months": 12,
    "services": ["Phone", "Internet"],
    "contract_type": "Month-to-month",
    "monthly_charges": 29.85,
    "total_charges": 358.2
  },
  "risk_assessment": {
    "current_score": 0.73,
    "risk_level": "High",
    "trend": "Increasing",
    "last_updated": "2025-11-22T10:30:00Z"
  },
  "history": [
    {
      "date": "2025-11-01",
      "score": 0.68,
      "action": "Model prediction"
    }
  ]
}
```

### Update Customer Profile

Update customer information.

```http
PUT /api/v1/customers/{customer_id}
Content-Type: application/json

{
  "monthly_charges": 34.99,
  "contract": "One year",
  "payment_method": "Credit card"
}
```

## Model Management

### Model Information

Get information about the current model.

```http
GET /api/v1/model/info
```

**Response:**
```json
{
  "model_version": "v2.1.0",
  "model_type": "XGBoost Ensemble",
  "training_date": "2025-11-15T08:00:00Z",
  "accuracy": 0.873,
  "precision": 0.852,
  "recall": 0.827,
  "f1_score": 0.839,
  "auc_roc": 0.914,
  "feature_count": 19,
  "training_samples": 7043
}
```

### Model Performance

Get detailed model performance metrics.

```http
GET /api/v1/model/performance
```

**Response:**
```json
{
  "metrics": {
    "accuracy": 0.873,
    "precision": 0.852,
    "recall": 0.827,
    "f1_score": 0.839,
    "auc_roc": 0.914
  },
  "confusion_matrix": {
    "true_positive": 421,
    "true_negative": 798,
    "false_positive": 73,
    "false_negative": 88
  },
  "feature_importance": {
    "contract": 0.189,
    "tenure": 0.156,
    "monthly_charges": 0.134
  },
  "drift_metrics": {
    "data_drift_score": 0.023,
    "concept_drift_score": 0.015,
    "last_check": "2025-11-22T06:00:00Z"
  }
}
```

## Business Analytics

### Risk Distribution

Get distribution of customers by risk level.

```http
GET /api/v1/analytics/risk-distribution
```

**Response:**
```json
{
  "total_customers": 7043,
  "risk_distribution": {
    "high_risk": {
      "count": 1058,
      "percentage": 15.0
    },
    "medium_risk": {
      "count": 2114,
      "percentage": 30.0
    },
    "low_risk": {
      "count": 3871,
      "percentage": 55.0
    }
  },
  "trends": {
    "high_risk_change": "+2.3%",
    "period": "last_30_days"
  }
}
```

### Feature Impact Analysis

Analyze the impact of features on churn predictions.

```http
GET /api/v1/analytics/feature-impact
```

**Response:**
```json
{
  "global_importance": {
    "contract": 0.189,
    "tenure": 0.156,
    "monthly_charges": 0.134,
    "total_charges": 0.098,
    "payment_method": 0.087
  },
  "feature_correlations": {
    "contract_tenure": 0.73,
    "charges_correlation": 0.89
  },
  "impact_analysis": [
    {
      "feature": "contract",
      "high_churn_value": "Month-to-month",
      "low_churn_value": "Two year",
      "impact_magnitude": 0.42
    }
  ]
}
```

## Monitoring & Metrics

### System Metrics

Get current system performance metrics.

```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "system": {
    "cpu_usage": 23.5,
    "memory_usage": 67.2,
    "disk_usage": 45.8,
    "uptime": 86400
  },
  "application": {
    "total_requests": 15847,
    "avg_response_time": 145,
    "error_rate": 0.002,
    "active_users": 23
  },
  "ml_metrics": {
    "predictions_served": 3421,
    "avg_prediction_time": 18,
    "model_accuracy": 0.873,
    "data_quality_score": 0.96
  }
}
```

### Alerts Status

Get current alert status and history.

```http
GET /api/v1/alerts
```

**Response:**
```json
{
  "active_alerts": [
    {
      "id": "ALERT_001",
      "severity": "warning",
      "message": "High prediction volume detected",
      "timestamp": "2025-11-22T09:45:00Z",
      "component": "prediction_service"
    }
  ],
  "resolved_alerts": [
    {
      "id": "ALERT_002",
      "severity": "critical",
      "message": "Model accuracy below threshold",
      "resolved_at": "2025-11-22T08:30:00Z"
    }
  ]
}
```

## Error Handling

All API endpoints return standard HTTP status codes and error responses.

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "monthly_charges",
      "issue": "Value must be positive"
    },
    "timestamp": "2025-11-22T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `MODEL_ERROR` | 500 | Model prediction failed |
| `SYSTEM_ERROR` | 500 | Internal system error |

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Prediction endpoints**: 100 requests per minute
- **Analytics endpoints**: 50 requests per minute
- **Management endpoints**: 20 requests per minute

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1732270260
```

## SDK and Client Libraries

### Python SDK

```python
from retentionai import RetentionAIClient

client = RetentionAIClient(
    base_url="https://retentionai.com/api/v1",
    api_key="your_api_key"
)

# Single prediction
prediction = client.predict(customer_data)
print(f"Churn probability: {prediction.churn_probability}")

# Batch prediction
predictions = client.predict_batch(customers_list)

# Get explanations
explanation = client.explain(customer_data)
```

### JavaScript SDK

```javascript
import { RetentionAI } from 'retentionai-js';

const client = new RetentionAI({
  baseURL: 'https://retentionai.com/api/v1',
  apiKey: 'your_api_key'
});

// Single prediction
const prediction = await client.predict(customerData);
console.log(`Churn probability: ${prediction.churnProbability}`);
```

## Webhooks

RetentionAI can send webhooks for important events:

### High Risk Customer Alert

Triggered when a customer's risk score exceeds threshold.

```json
{
  "event": "high_risk_customer",
  "customer_id": "CUST_001",
  "risk_score": 0.85,
  "previous_score": 0.65,
  "timestamp": "2025-11-22T10:30:00Z",
  "recommendations": [
    "Contact customer within 24 hours",
    "Offer retention incentives"
  ]
}
```

### Model Drift Detection

Triggered when model performance degrades.

```json
{
  "event": "model_drift_detected",
  "drift_score": 0.15,
  "threshold": 0.10,
  "affected_features": ["contract", "payment_method"],
  "recommendation": "Retrain model with recent data",
  "timestamp": "2025-11-22T10:30:00Z"
}
```

## Support

- **API Issues**: [api-support@retentionai.com](mailto:api-support@retentionai.com)
- **Documentation**: [docs.retentionai.com](https://docs.retentionai.com)
- **Status Page**: [status.retentionai.com](https://status.retentionai.com)
- **Community**: [Discord](https://discord.gg/retentionai)