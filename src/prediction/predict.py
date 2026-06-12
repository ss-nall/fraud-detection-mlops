import joblib
import pandas as pd

from src.utils.paths import (
    MODELS_DIR,
    PREDICTIONS_DIR
)


def run_prediction(
    input_file
):

    model = joblib.load(
        MODELS_DIR /
        "random_search_model.pkl"
    )

    threshold = joblib.load(
        MODELS_DIR /
        "optimal_threshold.pkl"
    )

    model_features = joblib.load(
        MODELS_DIR /
        "model_features.pkl"
    )

    df = pd.read_csv(
        input_file
    )

    X = df[
        model_features
    ].copy()

    fraud_probability = (

        model
        .predict_proba(
            X
        )[:, 1]

    )

    predicted_fraud = (

        fraud_probability
        >= threshold

    ).astype(int)

    predictions = pd.DataFrame({

        "fraud_probability":
            fraud_probability,

        "predicted_fraud":
            predicted_fraud

    })

    output_path = (

        PREDICTIONS_DIR /
        "predictions.csv"

    )

    predictions.to_csv(

        output_path,

        index=False

    )

    print(
        f"\nPredictions saved to:\n{output_path}"
    )

    return predictions


if __name__ == "__main__":

    run_prediction(
        "artifacts/preprocessing/X_test.csv"
    )