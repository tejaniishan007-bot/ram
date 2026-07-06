**Project Title:** Message Intelligence System
---

# 📖 Project Overview

The **Message Intelligence System** is a machine learning project developed to automatically classify digital messages as either **Spam** or **Legitimate**.

The project compares three different classification algorithms:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naïve Bayes

The objective is to understand the working principles of **distance-based**, **margin-based**, and **probabilistic** classifiers and evaluate their performance using various classification metrics.

---

# 🎯 Objectives

- Understand probability concepts in machine learning.
- Perform data preprocessing and feature engineering.
- Implement KNN classifier.
- Implement Support Vector Machine (SVM).
- Implement Gaussian Naïve Bayes classifier.
- Apply Bayes' Theorem manually.
- Compare classifier performance.
- Recommend the best model for spam detection.

---

# 📂 Dataset Information

The dataset contains message-related numerical features extracted from digital messages.

### Features

- Message Length
- Number of Special Characters
- Number of URLs
- Keyword Frequency Score
- Sender Activity Score
- Time-based Features
- Other numerical message characteristics

### Target Variable

| Label | Meaning |
|--------|----------|
| 0 | Legitimate Message |
| 1 | Spam Message |

---

# 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📁 Project Structure

```
Message Intelligence System/
│
├── Message Intelligence Dataset.csv
├── Message_Intelligence_System.ipynb
├── README.md
├── Project_Report.pdf
└── images/
```

---

# 📋 Project Workflow

## Part A
### Probability & Conceptual Foundation

- Conditional Probability
- Bayes' Theorem
- Naïve Bayes Assumption
- KNN
- SVM
- Classifier Comparison

---

## Part B
### Dataset Understanding & Preparation

- Load Dataset
- Dataset Information
- Missing Value Analysis
- Duplicate Checking
- Feature Selection
- Feature Scaling
- Train-Test Split

---

## Part C
### K-Nearest Neighbors (KNN)

- Model Building
- Different K Values
- Distance Metric Comparison
- Confusion Matrix
- ROC Curve
- Misclassified Messages

---

## Part D
### Support Vector Machine (SVM)

- Linear Kernel
- RBF Kernel
- Polynomial Kernel
- Support Vectors
- Kernel Comparison
- Comparison with KNN

---

## Part E
### Gaussian Naïve Bayes

- Model Training
- Probability Prediction
- Bayes' Theorem
- Manual Conditional Probability
- Manual Posterior Probability
- Comparison with Model Prediction

---

## Part F
### Model Comparison

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Comparison

- KNN
- SVM
- Naïve Bayes

---

## Part G
### Final Analysis

- Strengths of Models
- Weaknesses of Models
- Probability Assumptions
- Interpretability vs Performance
- Business Recommendation
- Final Conclusion

---

# 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

# 🤖 Machine Learning Models

## 1. K-Nearest Neighbors

Distance-based classifier that predicts the class based on the nearest neighboring samples.

---

## 2. Support Vector Machine

Margin-based classifier that finds the optimal separating hyperplane.

---

## 3. Gaussian Naïve Bayes

Probabilistic classifier based on Bayes' Theorem and Gaussian distribution.

---

# 📈 Results

The project compares all three models based on:

- Classification Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best-performing model is selected based on overall performance.

---

# 💼 Business Recommendation

For real-world spam detection systems:

- Use **Support Vector Machine (SVM)** when maximum classification accuracy is required.
- Use **Gaussian Naïve Bayes** when prediction speed and computational efficiency are more important.
- Use **KNN** for educational purposes or smaller datasets due to its simplicity.

---

# 📚 Learning Outcomes

After completing this project, the following concepts are demonstrated:

- Data Preprocessing
- Feature Scaling
- Train-Test Split
- Distance-Based Learning
- Margin-Based Learning
- Probabilistic Learning
- Bayes' Theorem
- Model Evaluation
- Model Comparison
- Business Recommendation

---

# ▶️ How to Run the Project

1. Clone or download the repository.
2. Open the project folder.
3. Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook
```

5. Run all notebook cells sequentially.

---

# 📦 Required Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
```

---

# 📸 Output

The notebook includes:

- Dataset Analysis
- Correlation Heatmap
- Feature Distribution
- Confusion Matrices
- ROC Curves
- Accuracy Comparison
- Precision Comparison
- Recall Comparison
- F1 Score Comparison
- Final Model Comparison

---

# ✅ Conclusion

This project successfully demonstrates the implementation and comparison of three machine learning classification algorithms for spam message detection.

Through preprocessing, model training, evaluation, and probability analysis, the project shows how different machine learning techniques solve the same classification problem.

Based on the evaluation metrics, the best-performing model is recommended for deployment in a real-world Message Intelligence System.


# ⭐ Thank You