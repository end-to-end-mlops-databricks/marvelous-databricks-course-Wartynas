import logging

import mlflow
import pandas as pd
import yaml
from databricks.connect import DatabricksSession
from mlflow.models import infer_signature

from diabetes.data_processor import DataProcessor
from diabetes.diabetes_model import DiabetesModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = DatabricksSession.builder.getOrCreate()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load configuration
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

logger.info("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))


# Initialize DataProcessor
data_processor = DataProcessor("data/diabetes_multiclass.csv", config)
data_processor.save_raw_data_to_catalog(spark)
logger.info("DataProcessor initialized and raw data saved to catalog.")

# Preprocess the data
df_transformed = data_processor.preprocess_data()
logger.info("Data preprocessed.")

# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()
logger.info("Data split into training and test sets.")
logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

train = pd.concat([X_train, y_train], axis=1)
train.columns = list(X_train.columns) + ["target"]

test = pd.concat([X_test, y_test], axis=1)
test.columns = list(X_test.columns) + ["target"]

data_processor.pandas_df_to_delta(df=train, name="train", spark=spark)

data_processor.pandas_df_to_delta(df=test, name="test", spark=spark)

logger.info("Train and test sets saved as delta tables.")

mlflow.set_experiment(experiment_name="/Workspace/Users/martynas.venckus@telesoftas.com/mlops_dev/model-training")
git_sha = "ffa63b430205ff7"

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id
    model = DiabetesModel(config)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    acc, f1 = model.evaluate(X_test, y_test)
    print(f"Accuracy: {acc}")
    print(f"F1 score: {f1}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "XGBoost classifier without preprocessing")
    mlflow.log_params(config["model_parameters"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1 score", f1)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_pandas(train, targets="target")
    mlflow.log_input(dataset, context="training")

    mlflow.pyfunc.log_model(python_model=model, artifact_path="xgboost-model", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/xgboost-model",
    name=f"{config['catalog_name']}.{config['schema_name']}.xgboost_model_basic",
    tags={"git_sha": f"{git_sha}"},
)
