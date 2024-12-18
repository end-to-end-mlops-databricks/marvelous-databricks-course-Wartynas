import numpy as np

from src.diabetes.data_processor import DataProcessor


def test_load_data(data_processor, sample_data):
    assert data_processor.df.equals(sample_data)


def test_preprocess_data(data_processor):
    data_processor.preprocess_data()

    assert data_processor.X.shape == (5, 4)
    assert data_processor.y.shape == (5,)
    assert set(data_processor.X.columns) == set(["num1", "num2", "cat1", "cat2"])
    assert data_processor.preprocessor is not None


def test_split_data(data_processor):
    data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data(test_size=0.4, random_state=42)

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 2


def test_preprocessor_transform(data_processor):
    data_processor.preprocess_data()
    X_transformed = data_processor.preprocessor.fit_transform(data_processor.X)

    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] > 4  # Due to one-hot encoding, we expect more columns


def test_missing_target(tmp_path, sample_data, sample_config):
    sample_data.loc[2, "target"] = np.nan
    csv_path = tmp_path / "test_data_missing.csv"
    sample_data.to_csv(csv_path, index=False)

    processor = DataProcessor(csv_path, sample_config)
    processor.preprocess_data()

    assert processor.df.shape[0] == 4  # One row should be removed due to missing target
