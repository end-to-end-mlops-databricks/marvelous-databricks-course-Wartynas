import pytest
import pandas as pd
from diabetes.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'cat1': ['A', 'B', 'C', 'A', 'B'],
        'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [10, 20, 30, 40, 50]
    })

@pytest.fixture
def sample_config():
    return {
        'num_features': ['num1', 'num2'],
        'cat_features': ['cat1', 'cat2'],
        'target': 'target'
    }

@pytest.fixture
def data_processor(tmp_path, sample_data, sample_config):
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, sample_config)