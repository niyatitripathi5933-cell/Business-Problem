import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor

def test_data_loading():
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    assert not df.empty
    assert len(df) == 300
    assert 'Price' in df.columns

def test_feature_engineering():
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    df_eng = preprocessor.feature_engineering(df)
    assert 'area_per_bedroom' in df_eng.columns
    assert 'age_category' in df_eng.columns
    assert len(df_eng) == len(df)

def test_preprocessor_pipeline():
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, preproc = preprocessor.preprocess_and_split()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]

if __name__ == '__main__':
    pytest.main([__file__])
