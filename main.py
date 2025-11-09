import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import create_pipeline, create_preprocessor, load_data

logging.basicConfig(level=logging.INFO)
logg = logging.getLogger(__name__)

DATA_PATH = Path("data/credit_risk_dataset_cleaned.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "credit_risk_experiment"
TARGET_COLUMN = "loan_status"

THRESHOLD = 0.30  # seuil pour la classification (à ajuster selon les besoins)


def main():
    logging.info("Loading data...")
    df = load_data(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN], axis=1)
    y = df[[TARGET_COLUMN]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        logging.info("Creating preprocessor...")
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        ordinal_cols = ["loan_grade"]
        nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

        preprocessor = create_preprocessor(numerical_cols, ordinal_cols, nominal_cols)

        logging.info("Creating pipeline...")

        classifier = RandomForestClassifier(random_state=RANDOM_STATE)
        pipeline = create_pipeline(preprocessor, classifier)

        logging.info("Training model...")
        pipeline.fit(X_train, y_train.values.ravel())

        logging.info("Evaluating model...")
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= THRESHOLD).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1-score: {f1}")

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("smote applied", True)
        mlflow.log_param("threshold", THRESHOLD)

        mlflow.log_metric("precision", precision)  # type: ignore
        mlflow.log_metric("recall", recall)  # type: ignore
        mlflow.log_metric("f1_score", f1)  # type: ignore

        report = classification_report(y_test, y_pred)
        logging.info(f"Classification Report:\n{report}")

        mlflow.log_text(report, "classification_report.txt")  # type: ignore
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="credit_risk_model",
            registered_model_name="CreditRiskModel",
        )

        joblib.dump(pipeline, "credit_risk_model.joblib")
        logging.info("Model saved as credit_risk_model.joblib")


if __name__ == "__main__":
    main()
