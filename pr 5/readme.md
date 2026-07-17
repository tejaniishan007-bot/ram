# Smart Outcome Predictor - Ensemble Learning Practical Exam

## Information of Exam

**Project:** Smart Outcome Predictor

**Dataset:** Smart Outcome Predictor Dataset.csv

---

# Project Overview

The objective of this practical exam is to build, train and evaluate multiple Ensemble Learning algorithms for both Classification and Regression problems.

The project predicts:

### Classification
- Completion Status

### Regression
- Final Score

The project demonstrates complete Machine Learning workflow from data preprocessing to model comparison.

---

# Dataset Information

Target Variables

Classification Target
- completion_status

Regression Target
- final_score

Input Features

- student_id
- age
- country_region
- device_type
- education_background
- course_level
- course_category
- course_start_date
- week_of_year
- sessions
- time_spent_hours
- videos_watched
- quiz_attempts
- assignments_submitted
- forum_posts
- avg_quiz_score
- attendance_rate

---

# Project Structure

Part A
- Dataset Loading
- Dataset Inspection

Part B
- Dataset Understanding
- Feature Selection
- Train-Test Split
- Data Preprocessing
- Missing Value Handling
- Encoding
- Feature Scaling

Part C
Bagging
- Decision Tree Classifier
- Bagging Classifier
- Decision Tree Regressor
- Bagging Regressor
- Performance Comparison

Part D
Boosting

Classification
- AdaBoost Classifier
- Gradient Boosting Classifier
- LightGBM Classifier
- XGBoost Classifier

Regression
- AdaBoost Regressor
- Gradient Boosting Regressor
- LightGBM Regressor
- XGBoost Regressor

Part E
Advanced Ensemble Learning

Classification
- Voting Classifier
- Stacking Classifier

Regression
- Voting Regressor
- Stacking Regressor

Part F
Model Evaluation

- Accuracy
- MAE
- RMSE
- R² Score

Model Comparison

Part G
Final Report

- Best Classification Model
- Best Regression Model
- Final Conclusion

---

# Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (

    BaggingClassifier,
    BaggingRegressor,

    AdaBoostClassifier,
    AdaBoostRegressor,

    GradientBoostingClassifier,
    GradientBoostingRegressor,

    RandomForestClassifier,
    RandomForestRegressor,

    VotingClassifier,
    VotingRegressor,

    StackingClassifier,
    StackingRegressor

)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.metrics import *

import matplotlib.pyplot as plt
```

Additional Libraries

```python
pip install lightgbm
pip install xgboost
```

---

# Data Preprocessing

The preprocessing pipeline consists of

1. Missing Value Imputation

Numerical Columns

- Median Imputation

Categorical Columns

- Most Frequent Imputation

2. Feature Scaling

- StandardScaler

3. Encoding

- OneHotEncoder

4. ColumnTransformer

---

# Ensemble Algorithms Used

## Bagging

- Decision Tree
- Bagging Classifier
- Bagging Regressor

Advantages

- Reduces Variance
- Reduces Overfitting
- Improves Stability

---

## Boosting

Algorithms

- AdaBoost
- Gradient Boosting
- LightGBM
- XGBoost

Advantages

- High Accuracy
- Sequential Learning
- Error Correction
- Better Generalization

---

## Voting

Classification

Hard Voting

Regression

Average Prediction

---

## Stacking

Base Models

- Decision Tree
- Random Forest
- Gradient Boosting

Meta Model

Classification

- Logistic Regression

Regression

- Linear Regression

---

# Evaluation Metrics

Classification

- Accuracy
- Confusion Matrix
- Classification Report

Regression

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Visualizations

The project generates

- Accuracy Comparison
- RMSE Comparison
- R² Comparison
- Model Comparison Charts

---

# Results

Classification

The model with the highest Accuracy is selected as the Best Classification Model.

Regression

The model with the highest R² Score and lowest MAE/RMSE is selected as the Best Regression Model.

---

# Conclusion

This project demonstrates:

✔ Complete Data Preprocessing

✔ Feature Engineering

✔ Ensemble Learning

✔ Bagging

✔ Boosting

✔ Voting

✔ Stacking

✔ Model Evaluation

✔ Performance Comparison

✔ Final Model Selection

The best-performing ensemble model is selected based on evaluation metrics and comparative analysis.

---

# Project Workflow

Dataset

↓

Data Cleaning

↓

Missing Value Imputation

↓

Encoding

↓

Feature Scaling

↓

Train-Test Split

↓

Bagging

↓

Boosting

↓

Voting

↓

Stacking

↓

Evaluation

↓

Comparison

↓

Final Report

---
