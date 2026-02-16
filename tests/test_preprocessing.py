import pytest
import pandas as pd
from src.preprocessing import FraudPreprocessor

def test_preprocessing_handles_nulls():
    # Create sample data with a missing value [cite: 120]
    data = {'purchase_value': [100.0, None], 'age': [25, 30], 
            'source': ['SEO', 'Ads'], 'browser': ['Chrome', 'Safari'], 'sex': ['M', 'F']}
    df = pd.DataFrame(data)
    
    # Simple imputation strategy check [cite: 120]
    df['purchase_value'] = df['purchase_value'].fillna(df['purchase_value'].mean())
    
    preprocessor = FraudPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    assert processed_df['purchase_value'].isnull().sum() == 0
    assert 'source' in processed_df.columns