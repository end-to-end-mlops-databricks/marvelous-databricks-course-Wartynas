from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, file_path:str, config):
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
        self.X = self.df[self.config['num_features'] + self.config['cat_features']]
        self.y = self.df[target_col]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ('cat', categorical_transformer, self.config['cat_features'])
            ])
        
    
    def split_data(self):
        return train_test_split(self.X, self.y, test_size=float(self.config["test_size"]), random_state=int(self.config["seed"]))






