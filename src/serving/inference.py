import json
import joblib
import pandas as pd

from src.utils.paths import (
    MODELS_DIR,
    INFERENCE_DIR
)

from src.transformation.temporal_features import (
    create_temporal_features
)

from src.transformation.velocity_features import (
    create_velocity_features
)

from src.transformation.amount_features import (
    create_amount_features
)

from src.transformation.behavior_features import (
    create_behavior_features
)

from src.transformation.device_location_features import (
    create_device_location_features
)


def run_inference(
    input_file
):

    if str(input_file).endswith(".csv"):

        df = pd.read_csv(
            input_file
        )

    elif str(input_file).endswith(".json"):

        df = pd.read_json(
            input_file
        )

    else:

        raise ValueError(
            "Only CSV and JSON files are supported."
        )

    original_df = df.copy()

    df = create_temporal_features(df)

    df = create_velocity_features(df)

    df = create_amount_features(df)

    df = create_behavior_features(df)

    df = create_device_location_features(df)

    drop_columns = [

        "transaction_id",

        "timestamp",

        "sender_id",

        "receiver_id",

        "device_id"

    ]

    existing_columns = [

        col

        for col in drop_columns

        if col in df.columns

    ]

    df = df.drop(
        columns=existing_columns
    )

    df = pd.get_dummies(

        df,

        columns=[
            "transaction_type"
        ],

        drop_first=False

    )

    df = df.fillna(0)

    model_features = joblib.load(

        MODELS_DIR /
        "model_features.pkl"

    )

    for feature in model_features:

        if feature not in df.columns:

            df[feature] = 0

    df = df[
        model_features
    ]

    # ============================================
    # LOAD MODEL
    # ============================================

    random_search_model = (

        MODELS_DIR /

        "random_search_model.pkl"

    )

    baseline_model = (

        MODELS_DIR /

        "baseline_model.pkl"

    )

    if random_search_model.exists():

        model = joblib.load(

            random_search_model

        )

        print(
            "Using Random Search Model"
        )

    else:

        model = joblib.load(

            baseline_model

        )

        print(
            "Using Baseline Model"
        )

    # ============================================
    # LOAD THRESHOLD
    # ============================================

    threshold_file = (

        MODELS_DIR /

        "optimal_threshold.pkl"

    )

    if threshold_file.exists():

        threshold = joblib.load(

            threshold_file

        )

    else:

        threshold = 0.5

        print(
            "Threshold file not found. Using default threshold = 0.5"
        )

    probabilities = model.predict_proba(
        df
    )[:, 1]

    predictions = (

        probabilities
        >= threshold

    ).astype(int)

    results = original_df.copy()

    results[
        "fraud_probability"
    ] = probabilities

    results[
        "predicted_fraud"
    ] = predictions

    results.to_csv(

        INFERENCE_DIR /
        "inference_predictions.csv",

        index=False

    )

    summary = {

        "total_transactions":
            int(len(results)),

        "fraud_predictions":
            int(
                predictions.sum()
            ),

        "legitimate_predictions":
            int(
                len(results)
                -
                predictions.sum()
            ),

        "fraud_percentage":
            float(
                predictions.mean()
                * 100
            ),

        "average_fraud_probability":
            float(
                probabilities.mean()
            ),

        "threshold":
            float(
                threshold
            )

    }

    with open(

        INFERENCE_DIR /
        "inference_summary.json",

        "w"

    ) as f:

        json.dump(

            summary,

            f,

            indent=4

        )

    print(
        f"\nLoaded {len(results)} transactions"
    )
    print(
        f"Frauds Detected: {predictions.sum()}"
    )
    print(
        f"Legitimate Transactions: {len(results) - predictions.sum()}"
    )
    print(
        f"Fraud Percentage: {summary['fraud_percentage']:.2f}%"
    )
    print(
        f"Threshold Used: {threshold}"
    )

    return {

        "predictions":
            results,

        "summary":
            summary

    }


if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage: python -m src.serving.inference <csv_or_json_file>"
        )

    else:

        output = run_inference(
            sys.argv[1]
        )

        print(
            json.dumps(
                output["summary"],
                indent=4
            )
        )
