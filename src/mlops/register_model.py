import mlflow
import mlflow.sklearn
import joblib

from src.utils.paths import (
    MODELS_DIR
)


def register_model():

    mlflow.set_experiment(
        "Fraud Detection"
    )

    with mlflow.start_run(
        run_name="model_registry"
    ):

        model = joblib.load(

            MODELS_DIR /
            "random_search_model.pkl"

        )

        mlflow.sklearn.log_model(

            sk_model=model,

            artifact_path="model",

            registered_model_name=
            "FraudDetectionModel"

        )

        print(
            "\nModel Registered Successfully\n"
        )


if __name__ == "__main__":

    register_model()