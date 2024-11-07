import pandas as pd
from pandas import read_csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataProcessor:
    def __init__(self, file_path: str, config):
        self.df = self.load_data(file_path)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    def load_data(self, path):
        return read_csv(path)

    def preprocess_data(self):
        target_col = self.config["target"]
        self.df = self.df.dropna(subset=[target_col])

        # Separate features and target
        self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        self.y = self.df[target_col]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("cat", categorical_transformer, self.config["cat_features"]),
            ]
        )

    def split_data(self, test_size=None, random_state=None):
        if test_size is None:
            test_size=self.config["test_size"]
        
        if random_state is None:
            random_state=self.config["seed"]

        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config['catalog_name']}.{self.config['schema_name']}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config['catalog_name']}.{self.config['schema_name']}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
