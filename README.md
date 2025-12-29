# Improved_detection_of_fraud_cases
#### Data Analysis, Cleaning, Feature Engineering, and Model Explainability

## Overview
This project focuses on detecting fraudulent transactions using two datasets, performing comprehensive data cleaning, feature engineering, modeling, and model explainability. The workflow covers exploratory data analysis (EDA), handling class imbalance, training baseline and ensemble models, and interpreting model predictions using SHAP.

---

## Data Preparation

### Fraud Dataset
- **Data Cleaning:** handled missing values, removed duplicates, corrected datatypes.
- **EDA:** analyzed univariate and bivariate distributions, visualized class imbalance, and quantified fraud patterns.
- **Geolocation Integration:** converted IP addresses to integers and mapped them to countries for fraud distribution analysis.
- **Feature Engineering:** created time-based features (`time_since_signup`, `hour_of_day`, `day_of_week`) and transaction frequency features (`user_transaction_count`, `avg_time_between_tx`).
- **Data Transformation:** scaled numerical features and encoded categorical features.
- **Class Imbalance Handling:** applied undersampling and SMOTE-like oversampling to training data.
- **Output:** cleaned, feature-rich dataset saved as `fraud_cleaned.csv`.

### Credit Card Dataset
- **Data Cleaning:** removed duplicates, verified missing values and datatypes.
- **EDA:** visualized class distribution, transaction amounts, and time features; performed correlation analysis on anonymized PCA features.
- **Class Imbalance Awareness:** noted extreme imbalance (<0.2% fraud), handled during model training.
- **Output:** cleaned dataset saved as `creditcard_cleaned.csv`.

---

## Modeling and Evaluation

### Data Preparation
- Split datasets using stratified train-test split.
- Separated features from target (`class`) and dropped raw datetime and high-cardinality ID columns (`signup_time`, `purchase_time`, `device_id`, `ip_address`).

### Models
1. **Baseline Model:** Logistic Regression with class balancing, trained on undersampled data.
2. **Ensemble Model:** Random Forest with hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`) using Grid Search and Stratified K-Fold cross-validation (k=5).
- **Preprocessing:** numerical features scaled with StandardScaler; categorical features one-hot encoded.

### Evaluation
- Metrics: F1-score, AUC-PR, confusion matrices.
- Cross-validation used for stability assessment.
- Performance comparison led to **Random Forest** selection.
- Outputs: visualizations for F1-score, AUC-PR, precision-recall curves, confusion matrices.  
- Best model saved as `models/best_random_forest.pkl`.

---

## Model Selection Justification

The **Random Forest model** was selected for deployment due to:

1. **Superior Performance Metrics:** Higher F1-score and AUC-PR than Logistic Regression.
2. **Robustness to Class Imbalance:** Ensemble approach handles imbalanced data effectively with stable cross-validation results.
3. **Ability to Capture Non-linear Patterns:** Can model complex feature interactions common in fraud.
4. **Interpretability & Practicality:** Provides feature importance insights; saved as `best_random_forest.pkl`.

---

## Model Explainability using SHAP

### Objective
Interpret the Random Forest model’s predictions using **SHAP** to identify the key drivers of fraud and derive actionable business recommendations.

### Key Steps
1. **Global Feature Importance**
   - Extracted built-in Random Forest feature importance.
   - Generated **top 10 feature importance plot**.
   - Created SHAP summary plot to identify features contributing most to predictions.

2. **Individual Predictions**
   - Generated SHAP force plots for:
     - True Positive (correctly identified fraud)
     - False Positive (legitimate flagged as fraud)
     - False Negative (missed fraud)
   - Highlighted features driving each specific prediction.

3. **Comparison & Interpretation**
   - Compared SHAP importance with Random Forest built-in importance.
   - Identified top 5 SHAP-driven features for fraud:
     1. `num__time_since_signup`  
     2. `cat__source_Direct`  
     3. `num__day_of_week`  
     4. `cat__source_SEO`  
     5. `num__user_id`  
   - Observed non-linear effects and interaction features captured by SHAP but not by RF importance alone.

4. **Business Recommendations**
   - **⏱️ Early Transaction Verification:** Transactions within 2 hours of signup should undergo additional verification. (`num__time_since_signup`)  
   - **🌐 Source Monitoring:** Transactions from Direct or SEO sources should be flagged for review. (`cat__source_Direct`, `cat__source_SEO`)  
   - **📅 Day-of-Week Alerts:** Increase monitoring on days with higher fraud likelihood. (`num__day_of_week`)  
   - **🆔 High-Risk User IDs:** Review users with unusual IDs triggering other risk factors. (`num__user_id`)

### Summary
- SHAP analysis complements Random Forest importance by revealing non-linear patterns and feature interactions.
- Provides actionable insights to guide fraud mitigation strategies in production.
- Ensures interpretability, transparency, and practical utility of the deployed model.

---
