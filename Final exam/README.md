# Customer Churn Prediction — Multi-Algorithm Classification Showdown

**Exam type:** Practical | **Duration:** 6 hours | **Role:** Junior Data Scientist, Telecom Retention Team

## 📌 Project Overview

A telecom company (similar to Jio or Airtel) is losing paying subscribers every month and cannot predict who is about to leave. This project builds and compares four supervised classification models to predict customer churn and recommends which one the retention team should deploy in production.

## 📂 Dataset

- **Name:** IBM Telco Customer Churn Dataset
- **Source:** [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 columns (demographics, services, billing, contract, churn)
- **Target:** `Churn` (Yes = churned, No = retained) — binary classification

## 🎯 Learning Objectives Covered

- Framing a churn prediction problem from a business and ML perspective (CAC vs. CLV, cost of false negatives)
- Exploratory Data Analysis on a real-world telecom dataset
- Implementation of KNN, Naive Bayes, SVM, and Decision Tree classifiers
- Handling class imbalance with `class_weight` and SMOTE
- Rigorous evaluation: confusion matrix, precision/recall trade-off, ROC-AUC
- Choosing the right model based on business cost of false negatives vs. false positives

## 🗂 Repository Structure

```
.
├── README.md                                  # this file
├── Customer_Churn_Prediction_SOLUTION.ipynb    # full solution notebook
├── Telco_Customer_Churn.csv                    # raw dataset (or link to Kaggle instead of committing raw data)
└── requirements.txt                            # Python dependencies
```

## ⚙️ Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn jupyter
jupyter notebook Customer_Churn_Prediction_SOLUTION.ipynb
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
jupyter
```

## 🧭 Notebook Walkthrough

| Step | Contents |
|---|---|
| 1 | Theory notes — churn/CAC/CLV, confusion matrix cells & business cost, class imbalance & SMOTE, how each algorithm works, precision vs. recall |
| 2 | EDA — load & inspect, fix `TotalCharges`, class-balance check, univariate & bivariate analysis (contract type, tenure buckets, monthly charges, correlations) |
| 3 | Preprocessing — drop `customerID`, handle nulls, feature engineering (`tenure_group`, `num_services`, `AutoPay`), binary/one-hot/label encoding, scaling, train-test split, SMOTE (fit on training fold only) |
| 4 | Model building — KNN, Naive Bayes, SVM, Decision Tree |
| 5 | Evaluation — confusion matrices, precision/recall/F1, ROC curves, precision-recall curves |
| 6 | Model comparison table & final recommendation for production deployment |
| 7 | Packaging notes for GitHub submission |

## 🔑 Key Findings

- The dataset is **imbalanced**: ~73.5% "No churn" vs. ~26.5% "Churn".
- **Month-to-month contracts**, **low tenure (0–12 months)**, and **higher MonthlyCharges** together form the highest-risk churn segment.
- Because a missed churner (**False Negative**) costs far more than an unnecessary retention offer (**False Positive**), the primary model-selection metric is **Recall** on the churn class, with **ROC-AUC** as a tie-breaker.

## 🏆 Model Recommendation

The model with the highest Recall (with a healthy ROC-AUC and acceptable Precision) from the comparison table in Step 6 of the notebook is recommended for deployment, scoring active customers on a recurring basis so the retention team can proactively target at-risk subscribers.

## 📝 Notes on Reproducibility

- `random_state=42` is used throughout (train-test split, SMOTE, models) for reproducible results.
- SMOTE is applied **only to the training set** — the test set remains the original, real-world imbalanced distribution to give an honest evaluation.
