from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path


def load_data(file_path):
    file_path = Path(file_path)
    data = pd.read_csv(file_path)
    return data


def create_preprocessor(numerical_cols, ordinal_cols, nominal_cols):
    numerical_transformer = StandardScaler()
    ordinal_transformer = OrdinalEncoder()
    nominal_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("ord", ordinal_transformer, ordinal_cols),
            ("nom", nominal_transformer, nominal_cols),
        ],
        remainder="passthrough",
    )
    return preprocessor


def create_pipeline(preprocessor, classifier: RandomForestClassifier):
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", classifier),
        ]
    )
    return pipeline
