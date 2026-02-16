import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FraudPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_cols = ['purchase_value', 'age'] # From Fraud_Data.csv [cite: 31, 36]
        self.categorical_cols = ['source', 'browser', 'sex'] # From Fraud_Data.csv [cite: 33, 34, 35]

    def fit_transform(self, df):
        # 1. Handle Categorical Encoding [cite: 140]
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # 2. Scale Numerical Features [cite: 139]
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        return df

    def transform(self, df):
        # Apply same transformations for inference (no fitting)
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col].astype(str))
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        return df