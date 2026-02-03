# Credit-Card-Fraud-Detection-Machine-Learning-Project
End-to-end credit card fraud detection project using machine learning. Performed deep EDA, feature engineering, and imbalance-aware modeling. Trained and tuned Random Forest and Gradient Boosting models using PR-AUC for reliable fraud detection.
Project Overview
This project builds an end-to-end fraud detection system using credit card transaction data. It includes Exploratory Data Analysis (EDA), feature engineering, model building, hyperparameter tuning, and deployment-ready inference. The core challenge addressed is extreme class imbalance, where fraudulent transactions account for less than 0.4% of all transactions. Therefore, precision-recall metrics and PR-AUC are emphasized over accuracy.

Objectives
- Perform in-depth EDA to understand transaction behavior
- Identify fraud patterns across time, geography, users, and merchants
- Engineer meaningful temporal, demographic, and distance-based features
- Train and evaluate multiple machine learning models
- Handle class imbalance using appropriate metrics and techniques
- Save and reuse trained models for inference

Dataset Description
- Total transactions: 555,719
- Target variable: is_fraud (0 = Non-Fraud, 1 = Fraud)
- Fraud rate: ~0.386%

Key features include transaction amount, category, merchant, user demographics, time-based attributes, geographic coordinates, and a distance feature calculated using the Haversine formula.

Exploratory Data Analysis (EDA)
EDA focused on:
- Data quality checks (no missing values or duplicates)
- Severe class imbalance analysis
- Temporal patterns (hour, day of week, month)
- Transaction amount distribution
- Category, merchant, job, city, and state-level fraud rates
- Distance-based fraud signals
- Correlation and point-biserial analysis

Key insights:
- Transaction amount is the strongest fraud indicator
- Fraud rates vary significantly by category, merchant, and time
- High-cardinality categorical features require careful encoding
- Distance is weak linearly but useful for tree-based models

Feature Engineering
- Extracted transaction hour, minute, and second
- Derived customer age from date of birth
- Calculated customer-merchant distance using the Haversine formula
- One-hot encoded categorical variables
- Standardized numerical features

Modeling Approach
Models trained:
- Logistic Regression
- Random Forest
- Gradient Boosting

Evaluation metrics:
- ROC-AUC
- Precision-Recall AUC
- Recall and precision for fraud class

Best baseline performance was achieved using Random Forest.

Hyperparameter Tuning
RandomizedSearchCV was used with stratified cross-validation, optimizing for average precision (PR-AUC). The tuned Random Forest improved recall while maintaining strong precision.

Final Model Performance
- ROC-AUC: ~0.99
- PR-AUC: ~0.69
- Fraud recall: ~79%

Model Deployment
The final model is saved using joblib and can be reused for inference on new transaction data.

Example output:
- Fraud probability score
- Binary fraud prediction

Tools and Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib

Data Source
Kaggle – Credit Card Transactions Fraud Detection dataset.
This is a synthetic dataset designed to simulate real-world credit card transactions for research and learning purposes.

Why This Project Matters
This project demonstrates strong analytical thinking, handling of imbalanced datasets, end-to-end machine learning workflow, business-relevant evaluation, and production-ready modeling.

Author
Manasi Khanolkar
Applied Mathematics & Data Analytics

Future Improvements
- Threshold optimization based on business cost
- Gradient boosting frameworks such as XGBoost or LightGBM
- Model explainability using SHAP
- Real-time fraud detection pipelines
