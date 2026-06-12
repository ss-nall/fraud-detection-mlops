import json
import joblib
import pandas as pd
from datetime import datetime

from src.utils.paths import (
    MODELS_DIR,
    EVALUATION_DIR,
    TUNING_DIR,
    MODEL_REGISTRY_DIR
)


def run_model_registry():

    with open(
        EVALUATION_DIR /
        "metrics.json",
        "r"
    ) as f:

        evaluation_metrics = json.load(
            f
        )

    with open(
        TUNING_DIR /
        "best_random_search_params.json",
        "r"
    ) as f:

        best_params = json.load(
            f
        )

    threshold = joblib.load(
        MODELS_DIR /
        "optimal_threshold.pkl"
    )

    features = joblib.load(
        MODELS_DIR /
        "model_features.pkl"
    )

    model_card = {

        "model_name":
            "LightGBM Fraud Detector",

        "version":
            "1.0.0",

        "created_at":
            datetime.now().isoformat(),

        "threshold":
            float(threshold),

        "feature_count":
            len(features),

        "features":
            list(features),

        "hyperparameters":
            best_params,

        "evaluation_metrics":
            evaluation_metrics

    }

    with open(
        MODEL_REGISTRY_DIR /
        "model_card.json",
        "w"
    ) as f:

        json.dump(
            model_card,
            f,
            indent=4
        )

    print(
        "\nModel Registry Created\n"
    )


if __name__ == "__main__":

    run_model_registry()