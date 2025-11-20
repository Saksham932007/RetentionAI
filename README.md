# RetentionAI 🧠📈

**Production-Grade Churn Prediction Application**

A comprehensive ML-powered platform for predicting customer churn and optimizing retention strategies using advanced machine learning, explainable AI, and business impact simulation.

## 🎯 Project Overview

RetentionAI is a full-stack data science application that helps businesses:
- Predict customer churn with high accuracy using XGBoost/CatBoost
- Understand churn drivers through SHAP explanations
- Simulate business impact with ROI calculations
- Generate personalized retention emails using AI
- Perform "What-If" analysis for retention strategies

## 🏗️ Architecture Roadmap

### Phase 1: Project Scaffolding & Database (Commits 1-8)
- [x] Git initialization, .gitignore, and README
- [ ] Directory structure creation
- [ ] Dependencies setup (pyproject.toml)
- [ ] Configuration management (src/config.py)
- [ ] Database layer with SQLAlchemy
- [ ] ETL pipeline skeleton
- [ ] Synthetic data generation
- [ ] Data ingestion to SQLite

### Phase 2: Data Engineering & Preprocessing (Commits 9-18)
- [ ] Data preprocessing framework
- [ ] Feature engineering pipelines
- [ ] Target encoding for categoricals
- [ ] Scaling and transformation
- [ ] Train/validation/test splits
- [ ] Utility functions and logging
- [ ] Preprocessing execution scripts
- [ ] Unit testing suite

### Phase 3: Modeling & Experiment Tracking (Commits 19-30)
- [ ] MLflow experiment setup
- [ ] Model training framework
- [ ] XGBoost implementation
- [ ] Hyperparameter optimization with Optuna
- [ ] Class imbalance handling
- [ ] Metrics tracking and artifacts
- [ ] Feature importance extraction
- [ ] Batch inference pipeline
- [ ] Model promotion system
- [ ] Training pipeline tests

### Phase 4: Advanced "Job Magnet" Logic (Commits 31-38)
- [ ] SHAP explainability integration
- [ ] Individual instance explanations
- [ ] ROI and business impact calculations
- [ ] Churn reduction simulation
- [ ] AI-powered email generation
- [ ] Counterfactual "What-If" analysis
- [ ] Business logic validation tests

### Phase 5: Streamlit Dashboard (Commits 39-48)
- [ ] Streamlit app foundation
- [ ] Interactive sidebar with system status
- [ ] Executive dashboard with KPIs
- [ ] Customer inspector with search
- [ ] SHAP visualization integration
- [ ] Simulation lab with sliders
- [ ] Real-time prediction updates
- [ ] Email generation interface
- [ ] Business impact calculator

### Phase 6: Deployment & Documentation (Commits 49-52)
- [ ] Docker containerization
- [ ] Docker Compose orchestration
- [ ] Comprehensive documentation
- [ ] Code quality and style fixes

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, SQLAlchemy, SQLite
- **ML/AI:** XGBoost, CatBoost, SHAP, Optuna, SMOTE
- **Experiment Tracking:** MLflow
- **Frontend:** Streamlit
- **Deployment:** Docker, Docker Compose
- **Data Science:** pandas, numpy, scikit-learn

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker (optional)

### Installation
```bash
git clone https://github.com/Saksham932007/RetentionAI.git
cd RetentionAI
pip install -r requirements.txt
```

### Usage
```bash
# Run the Streamlit application
streamlit run app.py

# Access the dashboard at http://localhost:8501
```

## 📊 Features

### 🎯 Core ML Features
- **Churn Prediction:** High-accuracy XGBoost models
- **Explainable AI:** SHAP-powered feature importance
- **Hyperparameter Tuning:** Optuna-based optimization
- **Class Imbalance Handling:** SMOTE and weighted algorithms

### 🔍 Business Intelligence
- **Executive Dashboard:** Real-time KPIs and metrics
- **Customer Inspector:** Individual customer risk profiles
- **Simulation Lab:** "What-If" scenario analysis
- **ROI Calculator:** Business impact quantification

### 🤖 AI-Powered Features
- **Retention Email Generation:** Personalized customer outreach
- **Counterfactual Analysis:** Intervention strategy simulation
- **Risk Segmentation:** Automated customer tier classification

## 📈 Performance Metrics

- **Model Accuracy:** >85% (target)
- **Precision/Recall:** Optimized for business impact
- **Feature Importance:** SHAP-based explanations
- **Response Time:** <200ms for predictions

## 🏢 Business Impact

### ROI Calculations
- **Customer Acquisition Cost (CAC):** Input parameter
- **Lifetime Value (LTV):** Calculated metric
- **Retention Cost:** Configurable intervention cost
- **Net Benefit:** Automated ROI computation

### Success Metrics
- Reduced churn rate by targeted interventions
- Improved customer retention strategies
- Data-driven decision making
- Automated personalization at scale

## 📚 Documentation

- **API Documentation:** Auto-generated from docstrings
- **Architecture Guide:** Detailed system design
- **Deployment Guide:** Production deployment steps
- **User Manual:** Dashboard usage instructions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Saksham Kapoor**
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]
- GitHub: [@Saksham932007](https://github.com/Saksham932007)

## 🙏 Acknowledgments

- MLflow for experiment tracking
- SHAP for explainable AI
- Streamlit for rapid dashboard development
- XGBoost for high-performance ML

---

*Built with ❤️ for data-driven customer retention*