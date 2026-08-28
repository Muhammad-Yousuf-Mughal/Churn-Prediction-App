# 📊 Customer Churn Prediction & Business Intelligence System

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://business-churn-prediction.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning and Business Intelligence solution designed to predict customer churn, identify key behavioral risk drivers, and deliver actionable retention insights through an interactive Web Dashboard.

🔗 **Live App:** https://business-churn-prediction.streamlit.app/

---

## 📌 Executive Summary

Acquiring new customers costs significantly more than retaining existing ones. This system processes customer behavior data, identifies key churn drivers, trains machine learning models, and translates predictive outputs into financial risk metrics for business stakeholders.

### Key Features
* **Predictive Risk Scoring:** Generates individual and batch churn probability scores.
* **Explainable AI (XAI):** Highlights root causes for customer risk (tenure, pricing, contract type, support friction).
* **Interactive Dashboard:** Deployed web application displaying executive KPIs, segment risks, and retention recommendations.

---

## 🎯 Business Problem & Objectives

Companies lose significant revenue when existing customers churn. This project addresses five core business questions:
1. Which customers are most likely to leave?
2. What structural or behavioral factors contribute to churn?
3. Which segments represent the highest financial exposure?
4. What is the total estimated revenue at risk?
5. How should retention teams prioritize interventions?

---

## 🏗️ Data Science Workflow

Raw Data ➔ Preprocessing ➔ Feature Engineering ➔ Model Benchmark ➔ Model Explainability ➔ Interactive STREAMLIT Dashboard

1. **Preprocessing:** Handling missing values, outlier detection, data scaling, and categorical encoding.
2. **Feature Engineering:** Creating domain-specific indicators (`Support_Calls_Per_Month`, `Payment_Delay_Rate`, `Average_Monthly_Spend`).
3. **Model Evaluation:** Benchmarking models against imbalanced class distributions using Precision, Recall, F1-Score, and ROC-AUC.
4. **Explainability:** Utilizing SHAP values and feature importance to avoid "black-box" outputs.
5. **Deployment:** Serving model predictions live via a Streamlit web application.

---

## 📊 Dataset Schema

| Category | Features | Description |
| :--- | :--- | :--- |
| **Customer** | Customer_ID, Age, Gender | Demographics and identification |
| **Account** | Account_Age, Contract_Type | Engagement duration and agreement structure |
| **Services** | Internet, Phone, Streaming | Subscribed product verticals |
| **Financial** | Monthly_Charges, Total_Charges | Financial commitment and spend |
| **Usage & Friction** | Monthly_Usage, Support_Calls, Payment_Delays | Platform interaction metrics |
| **Target** | Churn | Class Target (Yes: 1, No: 0) |

---

## 🤖 Model Performance Benchmark

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.79 | 0.63 | 0.54 | 0.58 | 0.83 |
| **Decision Tree** | 0.73 | 0.50 | 0.51 | 0.50 | 0.66 |
| **Random Forest** | 0.81 | 0.67 | 0.58 | 0.62 | 0.85 |
| **XGBoost / LightGBM** | **0.83** | **0.70** | **0.62** | **0.66** | **0.87** |

---

## 💡 Example Risk Output & Action Plan

Customer ID: CUST-10294
Churn Probability: 82.4% | Risk Level: HIGH

Top Risk Drivers:
1. Short customer tenure (< 3 months)
2. High monthly charges ($95/mo)
3. High support interaction frequency (> 4 calls/mo)
4. Month-to-month contract structure

Recommended Intervention:
Automate proactive support outreach and issue a 15% discount offer for a 12-month contract upgrade.

---

## 📂 Repository Structure

```text
customer-churn-project/
- data/
  - raw/
  - processed/
- notebooks/
  - 01_data_cleaning.ipynb
  - 02_eda.ipynb
  - 03_feature_engineering.ipynb
  - 04_model_training.ipynb
- src/
  - data_preprocessing.py
  - feature_engineering.py
  - train_model.py
  - evaluate_model.py
  - predict.py
- models/
  - churn_model.pkl
- dashboard/
  - app.py
- reports/
  - business_report.pdf
- requirements.txt
- README.md
- .gitignore
```

---



## ⚡ Quick Start & Local Setup

### Prerequisites
* Python 3.10+
* Git
---
### Commands
```bash
git clone https://github.com/Muhammad-Yousuf-Mughal/customer-churn-project.git
cd customer-churn-project
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Data & Analytics:** Pandas, NumPy, SQL
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Machine Learning & XAI:** Scikit-learn, XGBoost, SHAP
* **Deployment & UI:** Streamlit, Streamlit Cloud

---

## 📜 License

Distributed under the MIT License. See LICENSE for details.
