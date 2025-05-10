Insurance Fraud Detection using Machine Learning

Fraudulent insurance claims cost the industry billions of dollars annually. This project leverages supervised machine learning techniques to detect such claims and help insurers make data-driven decisions. Using classification models and real-world data, we aim to accurately distinguish between fraudulent and legitimate insurance claims.

Overview:

Problem: Detect and flag fraudulent insurance claims based on claim metadata
Dataset: 1,000+ real-world insurance claim records with mixed data types
Goal: Build and evaluate models that can predict whether a claim is fraudulent
Tech Stack: Python, Pandas, scikit-learn, XGBoost, Matplotlib, Seaborn

Dataset Details:

File: `insurance_claims.csv`
Target Column: `fraud_reported` (values: `Y` = Fraud, `N` = Not Fraud)
Features:

  * Demographics: `age`, `policy_deductable`, `policy_annual_premium`, etc.
  * Claim Details: `incident_type`, `collision_type`, `incident_severity`, etc.
  * Policyholder Info: `insured_occupation`, `insured_relationship`, etc.
  * Geographical Info: `incident_state`, `incident_city`, etc.

Exploratory Data Analysis (EDA):

Visualized class imbalance and key categorical variables
Explored correlations and feature distributions
Assessed null values and feature cardinality

Data Preprocessing:

Label encoding of the target variable (`fraud_reported`)
One-hot encoding for categorical features
Handled missing values and irrelevant columns
Performed an 80/20 train-test split

Model Training & Evaluation:

Implemented and compared the following classifiers:

| Model                    | Key Metrics Evaluated                |
| ------------------------ | ------------------------------------ |
| Logistic Regression      | Accuracy, Precision, Recall, F1      |
| Random Forest Classifier | Confusion Matrix, Feature Importance |
| XGBoost Classifier       | Best overall performance             |

Best Performing Model: **XGBoost**

Provided the most balanced precision and recall
Especially effective in reducing false negatives, critical for fraud detection

Visual Insights:

Confusion Matrices for all models
Classification Reports
Feature Correlation Heatmaps
Target Distribution Plot

Future Improvements:

Address class imbalance using SMOTE or class weights
Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
SHAP or LIME for model interpretability
Deploy the best model with a simple Flask or Streamlit app

Skills Demonstrated:

Data wrangling & EDA
Feature engineering & encoding
Binary classification & model evaluation
Confusion matrix and performance metric interpretation
End-to-end project pipeline in Jupyter Notebook

Author:
Deepakshi Mathur
