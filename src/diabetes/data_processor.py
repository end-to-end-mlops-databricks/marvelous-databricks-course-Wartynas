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
        self.y = self.df[self.config["target"]].astype(int)
        self.X = self.df[self.config["num_features"] + self.config["cat_features"]]
        self.X[self.config["cat_features"]] = self.X[self.config["cat_features"]].astype(str)
        self.X[self.config["num_features"]] = self.X[self.config["num_features"]].astype(int)

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

        self.X_transformed = self.preprocessor.fit_transform(self.X)
        self.X_df_transformed = pd.DataFrame(self.X_transformed, columns=self.preprocessor.get_feature_names_out())

        return self.X_df_transformed

    def split_data(self, test_size=None, random_state=None):
        if test_size is None:
            test_size = self.config["test_size"]

        if random_state is None:
            random_state = self.config["seed"]

        return train_test_split(self.X_df_transformed, self.y, test_size=test_size, random_state=random_state)

    def pandas_df_to_delta(self, df, name, spark):
        self._pandas_to_spark_to_delta_cdf(df, name, spark)

    def save_raw_data_to_catalog(self, spark: SparkSession):
        self._pandas_to_spark_to_delta_cdf(self.df, "raw_data", spark)

    def _pandas_to_spark_to_delta_cdf(self, pandas_df: pd.DataFrame, tbl_name: str, spark: SparkSession):
        spark_df = spark.createDataFrame(pandas_df).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        delta_table_path = f"{self.config['catalog_name']}.{self.config['schema_name']}.{tbl_name}"

        spark_df.write.mode("overwrite").saveAsTable(delta_table_path)

        spark.sql(f"ALTER TABLE {delta_table_path} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
