import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FraudPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing numerical values with the median[cite: 16]."""
        df = df.copy()
        df['purchase_value'] = df['purchase_value'].fillna(df['purchase_value'].median())
        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts categorical features into dummy variables[cite: 16]."""
        return pd.get_dummies(df, columns=['source', 'browser', 'sex'], drop_first=True)

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales numerical features to a 0-1 range[cite: 31]."""
        df = df.copy()
        df[['age']] = self.scaler.fit_transform(df[['age']])
        return df

    def align_features(self, df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
        """Ensures the dataframe matches the model's training schema[cite: 31]."""
        return df.reindex(columns=expected_columns, fill_value=0)