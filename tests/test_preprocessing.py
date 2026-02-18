import pytest
import pandas as pd
import numpy as np
from src.preprocessing import FraudPreprocessor

@pytest.fixture
def sample_data():
    """Provides a consistent dummy dataset for testing."""
    return pd.DataFrame({
        'purchase_value': [100.0, 200.0, np.nan],
        'age': [30, 40, 35],
        'source': ['Ads', 'SEO', 'Direct'],
        'browser': ['Chrome', 'Safari', 'Firefox'],
        'sex': ['M', 'F', 'M']
    })

def test_handle_missing_values(sample_data):
    """Test 1: Ensure missing purchase values are filled (Reliability)."""
    preprocessor = FraudPreprocessor()
    processed_df = preprocessor.clean_data(sample_data)
    assert processed_df['purchase_value'].isnull().sum() == 0

def test_categorical_encoding(sample_data):
    """Test 2: Verify categoricals are converted to numeric (Correctness)."""
    preprocessor = FraudPreprocessor()
    processed_df = preprocessor.encode_features(sample_data)
    
    # 'source_SEO' and 'source_Direct' exist because 'Ads' was dropped as the first category
    assert 'source_SEO' in processed_df.columns
    assert 'sex_M' in processed_df.columns
    assert processed_df['source_SEO'].dtype != object

def test_feature_scaling_range(sample_data):
    """Test 3: Verify numerical features are scaled (Standardization)."""
    preprocessor = FraudPreprocessor()
    processed_df = preprocessor.scale_features(sample_data)
    assert processed_df['age'].min() >= 0
    assert processed_df['age'].max() <= 1

def test_data_integrity_shape(sample_data):
    """Test 4: Verify row count remains unchanged (Data Integrity)."""
    preprocessor = FraudPreprocessor()
    processed_df = preprocessor.clean_data(sample_data)
    assert len(processed_df) == 3

def test_schema_alignment(sample_data):
    """Test 5: Verify reindexing handles missing columns (Robustness)."""
    preprocessor = FraudPreprocessor()
    expected = ['purchase_value', 'age', 'source_Ads', 'source_SEO']
    short_df = pd.DataFrame({'purchase_value': [50], 'age': [20], 'source_Ads': [1]})
    aligned_df = preprocessor.align_features(short_df, expected)
    assert 'source_SEO' in aligned_df.columns
    assert aligned_df['source_SEO'].iloc[0] == 0