# Contribution Margin Estimation with Machine Learning

## Project Overview

This project builds an **end-to-end machine learning pipeline** to estimate **Contribution Margin (CM)** for a dataset of approximately **15,000 ASINs**.

The objective is to **predict CM when it is missing** using correlated financial and operational variables.

The pipeline includes:

- Data collection
- Data preprocessing
- Feature engineering
- Exploratory clustering
- Predictive modeling with XGBoost
- Estimation of missing CM components

This project demonstrates a **full ML workflow applied to financial product performance analysis**.

---

# Project Structure
project-root
│
├── data
│ ├── raw
│ ├── processed
│ └── features
│
├── notebooks
│
├── src
│ ├── data_collection
│ ├── preprocessing
│ ├── feature_engineering
│ ├── clustering
│ ├── modeling
│ └── evaluation
│
├── models
├── reports
└── README.md


---

# Project Workflow

## 1. Data Collection

Collect product-level financial and operational metrics for ~15,000 ASINs.

Example variables:

- `revenue`
- `shipped_units`
- `product_cost_of_goods_sold`
- `display_ads_spend`
- `vendor_funding`
- `fulfillment_cost`
- `discounts`
- `net_profit`

Tasks:

- Extract data from source systems
- Store the raw dataset
- Validate schema and data types

Output:
data/raw/asins_financials.csv


---

# 2. Data Cleaning & Preprocessing

Prepare the dataset for modeling.

Steps:

- Handle missing values
- Remove duplicates
- Normalize numeric variables
- Standardize categorical features
- Convert data types
- Detect and cap outliers

Output:
data/processed/clean_dataset.parquet


---

# 4. Exploratory Clustering

Cluster ASINs to identify **similar economic structures between products**.

Algorithms explored:

- KMeans
- Hierarchical clustering
- DBSCAN (optional)

Purpose:

- Identify groups of products with similar cost structures
- Improve model understanding of product segments
- Use cluster labels as additional ML features

Outputs:

- Cluster labels
- Cluster feature profiles
- Cluster distribution plots

---

# 5. Model Training (XGBoost)

Train a regression model to predict **Contribution Margin (CM)**.

Model:
XGBoost Regressor

Target variable:
Contribution Margin (CM)


Features include:

- Revenue
- Units shipped
- Cost variables
- Engineered ratios
- Cluster labels

Training procedure:

- Train/test split
- Cross-validation
- Hyperparameter tuning

Evaluation metrics:

- RMSE
- MAE
- R²

Output:
models/xgboost_cm_model.pkl


---

# 6. Estimating Missing CM Components

Some components required to compute CM may be missing.

To address this, additional models estimate these components using **highly correlated variables**.

Examples:
COGS ≈ f(revenue, units, product_category)
Ads Spend ≈ f(revenue, category)
Discounts ≈ f(revenue, vendor_funding)


Models used:

- XGBoost
- Linear Regression
- ElasticNet

Goal:

Reconstruct missing variables required to compute CM.

---

# 7. CM Prediction Pipeline

Final pipeline:
Raw Data
↓
Preprocessing
↓
Feature Engineering
↓
Clustering
↓
XGBoost Prediction
↓
Estimated Contribution Margin


---

# 8. Model Evaluation

Evaluate prediction performance using:

- RMSE
- MAE
- MAPE
- R²

Additional analysis:

- Feature importance
- SHAP values
- Model performance across clusters

---

# 9. Results

Outputs include:

- Predicted **Contribution Margin per ASIN**
- Feature importance analysis
- Cluster segmentation insights
- Model performance metrics

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- NumPy
- Matplotlib / Seaborn
- Jupyter Notebooks

---

# Future Improvements
- Add time-series features
- Test additional models (LightGBM, CatBoost)
- Implement a production prediction pipeline
- Deploy model as an API
