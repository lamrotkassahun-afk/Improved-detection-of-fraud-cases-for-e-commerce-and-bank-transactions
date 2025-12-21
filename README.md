
# Fraud Detection for E-commerce and Bank Transactions

## 📌 Project Overview
[cite_start]This project, developed for Adey Innovations Inc., focuses on building advanced machine learning models to detect fraudulent activities in e-commerce and banking transactions [cite: 10-11]. [cite_start]By integrating geolocation analysis and transaction pattern recognition, the system aims to minimize financial losses while maintaining a seamless user experience[cite: 13, 16].

## 📂 Repository Structure
[cite_start]The repository is organized as follows to ensure a clean and professional workflow :
- [cite_start]`data/`: Contains raw and processed datasets (Note: Raw data is git-ignored) [cite: 89-92].
- [cite_start]`notebooks/`: Jupyter notebooks for EDA, Feature Engineering, and Modeling [cite: 93, 98-102].
- [cite_start]`scripts/`: Python scripts for modular data processing and utility functions[cite: 110].
- [cite_start]`src/`: Core source code for the project[cite: 104].
- [cite_start]`tests/`: Unit tests for ensuring code reliability[cite: 106].
- [cite_start]`models/`: Saved model artifacts (.pkl or .joblib)[cite: 108].

## 🛠️ Task 1: Data Analysis and Preprocessing
[cite_start]In this initial phase, we focused on preparing the `Fraud_Data.csv` and `IpAddress_to_Country.csv` datasets [cite: 116-117].

### Key Steps:
- [cite_start]**Data Cleaning**: Handled missing values and corrected data types for IP addresses and timestamps [cite: 119-122].
- [cite_start]**Geolocation Integration**: Converted IP addresses to integers and performed a range-based merge to map transactions to specific countries [cite: 127-130].
- **Feature Engineering**:
  - [cite_start]`time_since_signup`: Identified a high correlation between immediate purchases after signup and fraudulent activity[cite: 137].
  - [cite_start]`hour_of_day` & `day_of_week`: Captured temporal patterns of attackers [cite: 134-136].
- [cite_start]**Handling Class Imbalance**: Analyzed the extreme imbalance in the target variable and established a strategy to use **SMOTE** on the training set [cite: 141-144].

## 📈 Key Insights from EDA
- **Fraud Velocity**: Fraudulent transactions are significantly more likely to occur within the first few seconds of user signup.
- **Geographic Patterns**: The US and China show high transaction volumes, with specific fraud rates varying by region.

## 🚀 Future Work
- [cite_start]**Task 2**: Build and evaluate Logistic Regression and Ensemble models (XGBoost/Random Forest) using Stratified K-Fold cross-validation [cite: 145-166].
- [cite_start]**Task 3**: Utilize **SHAP** to provide model explainability and actionable business recommendations [cite: 167-186].

## ⚙️ Setup Instructions
1. Clone the repository: `git clone [Your Repo Link]`
2. [cite_start]Install dependencies: `pip install -r requirements.txt` [cite: 113, 236]
3. [cite_start]Explore the analysis in `notebooks/eda-fraud-data.ipynb`[cite: 98].
# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
A key challenge in fraud detection is managing the trade-off between security and user experience. False positives (incorrectly flagging legitimate transactions) can alienate customers, while false negatives (missing actual fraud) lead to direct financial loss. 
