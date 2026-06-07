# Fraud Detection for E-commerce and Bank Transactions

## Week 5 & 6 Challenge – 10 Academy AI Mastery Program

**Author:** Lalise Fufi
**Program:** 10 Academy – Artificial Intelligence Mastery Program
**Challenge Duration:** 4 Jun 2026 – 16 Jun 2026

---

# Project Overview

This project focuses on improving fraud detection for two different financial transaction datasets:

1. **E-commerce Transactions (Fraud_Data.csv)**
2. **Bank Credit Card Transactions (creditcard.csv)**

The objective is to develop machine learning models capable of accurately identifying fraudulent transactions while minimizing false positives and false negatives.

Fraud detection is a critical business problem because missed fraud cases result in financial losses, while incorrectly flagged legitimate transactions reduce customer trust and satisfaction.

---

# Business Objective

Adey Innovations Inc. aims to build a unified fraud detection system that can handle:

* E-commerce transactions containing user, device, behavioral, and geolocation information.
* Credit card transactions containing anonymized PCA-transformed features.

The solution should:

* Detect fraudulent transactions effectively.
* Handle severe class imbalance.
* Support explainable decision-making.
* Provide actionable business insights.

---

# Datasets

## 1. Fraud_Data.csv

E-commerce transaction dataset containing:

* User information
* Device information
* Browser information
* Transaction details
* IP addresses
* Fraud labels

Target Variable:

* `class`

  * 0 = Legitimate Transaction
  * 1 = Fraudulent Transaction

---

## 2. IpAddress_to_Country.csv

Contains IP address ranges and corresponding countries.

Used for geolocation enrichment.

---

## 3. creditcard.csv

Credit card transaction dataset containing:

* Time
* Amount
* PCA-transformed features (V1–V28)

Target Variable:

* `Class`

  * 0 = Legitimate Transaction
  * 1 = Fraudulent Transaction

---

# Project Structure

```text
fraud-detection-week5-6/
│
├── .github/
│   └── workflows/
│       └── unittests.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│
├── src/
│
├── tests/
│
├── models/
│
├── scripts/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Task 1: Data Analysis and Preprocessing

## Data Cleaning

The following preprocessing steps were performed:

### Missing Values

All datasets were inspected for missing values.

Results:

* Fraud_Data.csv → No missing values
* creditcard.csv → No missing values
* IpAddress_to_Country.csv → No missing values

No imputation was required.

---

### Duplicate Records

Duplicate analysis results:

| Dataset     | Duplicate Records |
| ----------- | ----------------- |
| Fraud_Data  | 0                 |
| IP Dataset  | 0                 |
| Credit Card | 1081              |

Duplicate transactions were removed from the credit card dataset to improve data quality.

---

### Data Type Conversion

Timestamp variables were converted to datetime format:

* signup_time
* purchase_time

IP addresses were converted to integer format for geolocation mapping.

---

# Exploratory Data Analysis (EDA)

## Fraud Dataset

### Fraud Distribution

Class distribution:

* Legitimate Transactions: 90.64%
* Fraudulent Transactions: 9.36%

The dataset is moderately imbalanced.

---

### Purchase Value Analysis

Findings:

* Purchase values are positively skewed.
* Most transactions involve relatively small purchase amounts.
* Several high-value outliers exist.

---

### Age Distribution

Findings:

* Most users are between 20 and 40 years old.
* Age distribution is approximately bell-shaped with slight right skewness.
* Very few users are above 60 years old.

---

### Fraud vs Purchase Value

Findings:

* Fraudulent and legitimate transactions show similar purchase value distributions.
* Purchase value alone is unlikely to be a strong fraud indicator.

---

### Browser Analysis

Findings:

* Chrome generated the largest number of transactions.
* Fraudulent transactions occur across all browser types.
* Browser information may provide useful supplementary predictive information.

---

## Credit Card Dataset

### Class Distribution

Class counts:

* Legitimate Transactions: 283,253
* Fraudulent Transactions: 473

Fraud cases represent approximately 0.17% of transactions.

This dataset exhibits extreme class imbalance.

---

### Transaction Amount Distribution

Findings:

* Most transactions involve small monetary amounts.
* Distribution is highly right-skewed.
* A small number of very large transactions are present.

---

### Fraud vs Transaction Amount

Findings:

* Fraudulent transactions generally involve smaller transaction amounts.
* Legitimate transactions contain a wider range of values and more extreme outliers.

---

# Geolocation Integration

## IP Address Mapping

IP addresses were mapped to countries using a range-based lookup process.

Steps:

1. Convert IP addresses to integers.
2. Sort datasets by IP range.
3. Perform range-based merge.
4. Validate IP ranges.
5. Assign countries.

---

## Fraud Analysis by Country

Countries with the highest observed fraud rates include:

* Turkmenistan
* Namibia
* Sri Lanka
* Luxembourg
* Virgin Islands (U.S.)
* Ecuador
* Tunisia
* Peru
* Bolivia
* Kuwait

### Key Insight

Fraud rates vary substantially across countries, indicating that geographic information may contribute meaningful predictive power.

Transaction volume analysis also revealed that:

* United States generated the highest number of transactions.
* China and Japan followed as the next largest transaction sources.

Country-level behavior will be retained as a modeling feature.

---

# Feature Engineering

Several fraud-related behavioral and temporal features were created.

## Engineered Features

### time_since_signup

Measures the elapsed time between account registration and purchase.

Purpose:

Fraudsters often make purchases shortly after creating accounts.

---

### hour_of_day

Extracted transaction hour (0–23).

Purpose:

Fraud activity may occur during unusual hours.

---

### day_of_week

Extracted transaction weekday.

Purpose:

Detect weekly behavioral patterns.

---

### transaction_count

Counts transactions performed by each user.

Purpose:

Identify unusually active users.

---

### device_frequency

Counts transactions associated with each device.

Purpose:

Detect devices reused across multiple transactions.

---

## Feature Engineering Insights

Summary statistics indicate:

* Average time since signup is approximately 4.9 million seconds.
* Transactions occur throughout all hours of the day.
* Device usage varies substantially, with some devices appearing up to 20 times.
* Transaction count remained relatively constant because most users made only a single recorded transaction.

These engineered features capture behavioral signals that are not directly available in the original data.

---

# Class Imbalance Analysis

## Fraud Dataset

* Legitimate Transactions: 90.64%
* Fraudulent Transactions: 9.36%

The dataset is moderately imbalanced.

---

## Credit Card Dataset

* Legitimate Transactions: 283,253
* Fraudulent Transactions: 473

Fraudulent transactions represent approximately 0.17% of all observations.

This severe imbalance requires special handling during model training.

---

## Planned Resampling Strategy

For Task 2:

* SMOTE (Synthetic Minority Over-sampling Technique) will be applied.
* Resampling will be performed only on the training set.
* Test data will remain untouched to prevent data leakage.

---

## Evaluation Metrics

Model performance will be evaluated using:

* Precision
* Recall
* F1-Score
* Precision-Recall AUC (AUC-PR)
* Confusion Matrix

Accuracy will not be used as the primary metric due to class imbalance.

---

# Interim-1 Deliverables Completed

✅ Data Cleaning

✅ Exploratory Data Analysis

✅ Geolocation Integration

✅ Feature Engineering

✅ Class Imbalance Analysis

---

# Next Steps (Task 2)

The next phase will focus on:

* Train-Test Split
* SMOTE Resampling
* Logistic Regression Baseline Model
* Random Forest / XGBoost Model
* Model Comparison
* Cross-Validation
* Performance Evaluation

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn

---

# Author

**Lalise Fufi**

10 Academy – Artificial Intelligence Mastery Program
