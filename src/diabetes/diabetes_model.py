from xgboost import XGBClassifier
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score


class DiabetesModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config):
        self.config = config
        self.model = XGBClassifier(
                objective=config['model_parameters']['obj_fn'],
                max_depth=config['model_parameters']['max_depth'],
                learning_rate=config['model_parameters']['learning_rate'],
                n_estimators=config['model_parameters']['n_estimators']
            )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, labels=[0,1,2], average="macro")
        return acc, f1

    # def get_feature_importance(self):
    #     feature_importance = self.model.named_steps['regressor'].feature_importances_
    #     feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
    #     return feature_importance, feature_names

