
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
# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## Project Overview
[cite_start]This project, developed for Adey Innovations Inc., focuses on building a robust end-to-end fraud detection system for e-commerce and banking sectors[cite: 10, 11]. [cite_start]The goal is to identify fraudulent activities accurately while balancing the trade-off between security and user experience (minimizing False Positives and False Negatives) [cite: 16-18].

## Key Features
- [cite_start]**Geolocation Integration**: Mapping IP addresses to countries to identify high-risk regions [cite: 127-130].
- [cite_start]**Feature Engineering**: Creation of time-based features (e.g., `time_since_signup`) and transaction velocity metrics [cite: 132-137].
- [cite_start]**Imbalance Handling**: Utilization of SMOTE to balance the highly skewed fraud datasets[cite: 141, 142].
- [cite_start]**Model Explainability**: Integration of SHAP and LIME to provide transparency into model decisions [cite: 167-172].
- [cite_start]**Real-time API**: A Flask-based REST API for immediate fraud prediction[cite: 20, 230].
- [cite_start]**Interactive Dashboard**: A Dash application for visualizing global fraud trends[cite: 229, 237].

## Project Structure
- [cite_start]`data/`: Raw and processed datasets[cite: 89].
- [cite_start]`models/`: Saved model artifacts (.pkl files)[cite: 108].
- [cite_start]`notebooks/`: Jupyter notebooks for EDA, Feature Engineering, Modeling, and Explainability[cite: 93].
- [cite_start]`src/`: Source code for the Flask API and Dash Dashboard[cite: 104].
- [cite_start]`scripts/`: Utility scripts for data processing[cite: 110].

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ## 🛠️ Machine Learning Pipeline (Refined)
Following industry best practices, the modeling phase now includes:
- **Baseline Establishment**: Logistic Regression serves as the performance floor.
- **Stratified K-Fold CV**: Ensures model reliability across imbalanced data subsets.
- **Ensemble Comparison**: Rigorous evaluation of Decision Trees, Random Forest, and Gradient Boosting.
- **Model Selection**: The final model was chosen based on AUC-PR (Area Under Precision-Recall Curve), which is superior for fraud detection than standard accuracy.
##Strategic Business Recommendations
1. Implement Velocity-Based Friction (New Account Protection)
Data Insight: SHAP analysis consistently identifies time_to_purchase (the duration between account creation and the first transaction) as a top predictor of fraud. Automated bots often register and purchase within seconds.

Recommendation: Implement a "New Account Cooling Period." Any transaction initiated within the first 10 minutes of account creation should be automatically flagged for secondary verification (e.g., SMS or Email OTP).

Business Impact: Reduces automated "flash" fraud attacks while maintaining a smooth experience for long-term users.

2. Multi-Account Device Throttling
Data Insight: High device_id frequency—where a single hardware ID is linked to multiple user accounts—is a significant red flag in the model.

Recommendation: Establish a Device-to-User Limit. Automatically trigger a manual security review or temporary account lock if a single device is used to access more than three unique user accounts within a 24-hour window.

Business Impact: Effectively mitigates "Account Takeover" (ATO) schemes and professional fraud farm operations.

3. Dynamic Geo-Risk Authentication
Data Insight: The interactive dashboard and choropleth maps highlight specific geographic regions with fraud rates 3x to 5x higher than the global average.

Recommendation: Deploy Location-Aware Step-Up Authentication. Transactions originating from high-risk IP addresses or countries should require Multi-Factor Authentication (MFA) for any purchase exceeding $50, compared to a $200 threshold for low-risk regions.

Business Impact: Localizes security measures, ensuring high-risk areas are tightly monitored without adding unnecessary friction to customers in stable markets.

4. Adaptive Purchase Value Monitoring
Data Insight: Local SHAP force plots reveal that purchase_value acts as a heavy "push" factor toward fraud when combined with specific browsers or times of day.

Recommendation: Use Behavioral Baseline Scoring. Instead of fixed limits, flag transactions that deviate by more than 2.5 standard deviations from a user's historical average or the average for that specific demographic group.

Business Impact: Captures "outlier" spending patterns that often indicate compromised credit card information.

Where to Place These in Your Project Structure
README.md: Include the titles and 1-sentence summaries under a ## 📈 Business Insights header.

Project Report/Blog: Use the full text above, including the "Data Insight" and "Business Impact" sections, to demonstrate your ability to bridge data science and business value.

Explainability Notebook: Add a final Markdown cell titled "Task 3.4: From SHAP to Strategy" and list these points to conclude your technical analysis.