# Credit Card Fraud Detection

## Overview
This project aims to classify credit card transactions as fraudulent or non-fraudulent using machine learning techniques. The dataset used for this project is the "Credit Card Fraud Detection" dataset from Kaggle, which contains anonymized features and transaction information.

## Problem Statement
The objective is to build a robust model capable of identifying fraudulent transactions to mitigate risks and improve financial security.

## Dataset
- **Source**: Kaggle - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description**: The dataset includes 284,807 transactions with 31 features (28 anonymized features, time, amount, and class).
- **Class Distribution**:
  - 0: Non-fraudulent transactions (majority class)
  - 1: Fraudulent transactions (minority class)

## Steps Followed

### 1. Data Preprocessing
- Checked for null values and data consistency.
- Explored class imbalance in the dataset.
- Visualized feature distributions.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of fraudulent and non-fraudulent transactions.
- Visualized transaction amount trends and time-based patterns.

### 3. Feature Engineering
- Applied standardization to numerical features.
- Addressed the class imbalance using oversampling techniques (e.g., SMOTE).

### 4. Model Development
- Implemented various machine learning models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost)
  - Neural Networks
- Evaluated performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

### 5. Model Evaluation
- Used cross-validation to assess model performance.
- Compared metrics to identify the best-performing model.

## Result
- Achieved high precision and recall on the minority (fraudulent) class.
- Selected the best-performing model based on AUC-ROC and F1-score.
