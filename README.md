# AI Powered Retail Inventory Optimization System

## Overview

The **AI Powered Retail Inventory Optimization System** is a machine learning–based decision support application developed to assist retailers in inventory planning, demand analysis, and pricing optimization. The system integrates data preprocessing, exploratory data analysis (EDA), machine learning, inventory analytics, and an interactive Streamlit dashboard into a single workflow.

The project demonstrates a complete machine learning pipeline, from raw retail data processing to business insights, enabling users to explore demand trends, monitor inventory levels, and analyze pricing strategies.

---

## Project Objectives

* Perform data cleaning and exploratory data analysis on retail datasets.
* Forecast product demand using machine learning techniques.
* Analyze inventory levels and identify stockout risks.
* Calculate safety stock and recommended reorder points.
* Compare product prices with competitor pricing.
* Generate pricing recommendations based on business metrics.
* Present results through an interactive Streamlit dashboard.

---

## Key Features

### Demand Forecasting

* Data preprocessing and feature engineering
* Demand prediction using supervised machine learning
* Actual vs. Predicted demand visualization
* Product-level and store-level demand analysis

**Models Implemented**

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Extra Trees Regressor
* XGBoost Regressor

**Selected Model**

```
models/demand_forecasting2_model.pkl
```

---

### Inventory Optimization

The inventory module performs:

* Stock level monitoring
* Low stock identification
* Stockout risk detection
* Warehouse utilization analysis
* Safety stock calculation
* Recommended reorder point generation

Generated output:

```
data/processed/final_inventory_optimization.csv
```

---

### Pricing Optimization

The pricing module provides:

* Competitor price comparison
* Price gap analysis
* Elasticity analysis
* Pricing recommendations

Generated output:

```
data/processed/pricing_optimized.csv
```

---

## Project Structure

```
Inventory-Optimization-System

├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── demand_forecasting.csv
│   │   ├── inventory_monitoring.csv
│   │   └── pricing_optimization.csv
│   │
│   └── processed/
│       ├── demand_cleaned.csv
│       ├── demand_forecast.csv
│       ├── inventory_cleaned.csv
│       ├── final_inventory_optimization.csv
│       ├── pricing_cleaned.csv
│       └── pricing_optimized.csv
│
├── models/
│   └── demand_forecasting2_model.pkl
│
├── notebooks/
│   ├── demand_forecasting.ipynb
│   ├── inventory_monitoring.ipynb
│   └── pricing_optimization.ipynb
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Plotly
* Streamlit

---

## Machine Learning Workflow

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Demand Prediction
      │
      ▼
Inventory Optimization
      │
      ▼
Pricing Recommendation
      │
      ▼
Interactive Dashboard
```

---

## Model Evaluation

The demand forecasting models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

**Selected Model:** Linear Regression

| Metric   |  Score |
| -------- | -----: |
| MAE      | 125.66 |
| RMSE     | 145.26 |
| R² Score | -0.002 |

> **Note:** The provided dataset appears to contain limited predictive relationships between the input features and the target variable. As a result, the trained models achieve a low R² score. This project primarily demonstrates the complete machine learning workflow and decision-support system rather than high predictive accuracy.

---

## Dashboard Modules

### Executive Dashboard

* Retail overview
* Total products and stores
* Demand summary
* Inventory alerts
* Pricing insights

### Demand Forecasting

* Demand prediction interface
* Actual vs. Predicted visualization
* Model performance metrics

### Inventory Optimization

* Stock monitoring
* Safety stock calculation
* Reorder point recommendations
* Inventory risk analysis

### Pricing Optimization

* Competitor price comparison
* Price gap visualization
* Pricing recommendations

---

## Installation

### Clone the repository

```bash
git clone <repository-url>
```

### Navigate to the project directory

```bash
cd Inventory-Optimization-System
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

* Integration with real-time retail sales data
* Time-series forecasting models (ARIMA, Prophet, LSTM)
* Dynamic pricing using reinforcement learning
* Cloud deployment
* Automated inventory alerts
* REST API integration
* Model retraining with live business data

---

## Author

**Shivam Subedar Patel**

AI/ML Intern

---

## Conclusion

The AI Powered Retail Inventory Optimization System demonstrates an end-to-end machine learning workflow for retail analytics. The project integrates demand forecasting, inventory optimization, and pricing analysis into an interactive dashboard, showcasing data preprocessing, feature engineering, model development, and business intelligence techniques. While the current dataset provides limited predictive performance, the system serves as a scalable foundation that can be extended with real-world retail data for improved forecasting and decision-making.
