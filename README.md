# medical_insurance_cost_prediction_advance_ml

## ** Summary**

This project demonstrates a full, industry-quality ML workflow for predicting medical insurance charges. It leverages advanced tree-based models, stacking, log-transform techniques, and SHAP interpretability to build a robust predictive system. The final CatBoost model achieves strong accuracy and provides transparent insights into the drivers of healthcare costs.

<img width="465" height="397" alt="image" src="https://github.com/user-attachments/assets/ea41d15a-03f5-42cc-8ef8-4075a6c22173" />

---

# **Medical Insurance Cost Prediction (Advanced ML Pipeline)**

Predicting healthcare insurance charges using advanced machine learning techniques.

This project builds a complete ML pipeline to model medical insurance costs based on demographic and lifestyle features. The workflow includes exploratory data analysis (EDA), preprocessing, multiple regression models, hyperparameter tuning, ensemble stacking, log-transform modeling, SHAP interpretability, and full residual diagnostics.

---

## ** Project Goal**

To build an end-to-end regression system that predicts an individual’s medical insurance charges using demographic and health-related variables. The focus is on model accuracy, interpretability, and industry-standard ML workflow practices.

---

## ** Dataset Overview**

Source: Medical Insurance Cost Dataset
Total rows: **1338**

### **Features**

* **age** — Age of primary beneficiary
* **sex** — Gender (male/female)
* **bmi** — Body Mass Index
* **children** — Number of dependents
* **smoker** — Smoking status (yes/no)
* **region** — Residential region (NE, NW, SE, SW)
* **charges** — Target variable; total medical cost billed

### **Common Uses**

* Regression modeling
* Insurance pricing research
* Health economics
* ML education and feature engineering
* Understanding cost drivers in healthcare

---

## ** Methodology**

### **1. Exploratory Data Analysis (EDA)**

* Distribution plots of features
* Pairplots and correlation heatmaps
* Boxplots of charges across categories
* Statistical tests (t-tests, correlation metrics)

### **2. Data Preprocessing**

* Train–test split
* StandardScaler for numeric features
* OneHotEncoder for categorical features
* ColumnTransformer pipeline
* Polynomial and interaction features

### **3. Models Implemented**

The project compares a wide set of models:

**Linear and Regularized Models**

* Linear Regression
* Ridge, Lasso, ElasticNet
* Polynomial Regression

**Tree-Based Models**

* Random Forest
* GradientBoostingRegressor
* HistGradientBoostingRegressor
* XGBoost
* LightGBM
* CatBoost

**Ensemble Techniques**

* Gradient Boosting with hyperparameter tuning
* Log-transform regression
* Quantile regression (95th percentile)
* Stacking ensemble (CatBoost + LightGBM + GBR with Ridge meta-learner)

**Explainability**

* SHAP summary plots
* SHAP dependence analysis

**Diagnostics**

* Residual vs. prediction plots
* Residual distribution
* QQ plots
* Error analysis

---

## **🏆 Final Model & Results**

After evaluating all models, **CatBoost** delivered the strongest performance.

### **Final Metrics**

| Model                     | RMSE      |
| ------------------------- | --------- |
| Linear Regression         | ~6100     |
| Gradient Boosting (Tuned) | ~4620     |
| LightGBM                  | ~5002     |
| Stacking Model            | ~4434     |
| **CatBoost (Best)**       | **~4426** |

### **Why CatBoost Won**

* Handles categorical data natively
* Captures nonlinear interactions automatically
* Strong performance on small tabular datasets
* Excellent bias-variance balance
* Stable and fast training
* Highly interpretable with SHAP

---

## **📊 SHAP Insights (Feature Importance)**

SHAP analysis revealed that:

* **Smoker** is by far the strongest predictor of insurance cost
* **Age** increases costs consistently
* **BMI** and **BMI²** capture obesity-related risk
* **Age × BMI** interactions matter
* **Region** has mild effect
* **Sex** and **children** have minimal impact

These patterns align with real-world medical cost dynamics.

---

## ** Residual Diagnostics**

Residual analysis shows:

* Good performance on low–medium charges
* Increasing error variance for high-cost patients (heteroscedasticity)
* Some underprediction on extreme medical bills
* Residuals are right-skewed, consistent with healthcare claims data
* QQ plot confirms heavy-tailed distribution

This behavior is expected in insurance forecasting tasks.

---

## ** Future Improvements**

Potential enhancements:

* Tweedie regression (industry standard for insurance)
* Optuna hyperparameter tuning
* CatBoost Bayesian optimization
* Conformal prediction intervals
* Partial Dependence Plots (PDP)
* Streamlit dashboard or FastAPI deployment

---

