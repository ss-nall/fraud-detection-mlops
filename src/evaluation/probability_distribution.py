import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import (
    PREPROCESSING_DIR,
    MODELS_DIR,
    EVALUATION_DIR
)


def run_probability_distribution():

    X_test = pd.read_csv(
        PREPROCESSING_DIR /
        "X_test.csv"
    )

    y_test = pd.read_csv(
        PREPROCESSING_DIR /
        "y_test.csv"
    ).squeeze()

    model_path = (
        MODELS_DIR /
        "random_search_model.pkl"
    )

    if not model_path.exists():

        model_path = (
            MODELS_DIR /
            "baseline_model.pkl"
        )

    model = joblib.load(
        model_path
    )

    threshold = joblib.load(
        MODELS_DIR /
        "optimal_threshold.pkl"
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    legit_probs = probabilities[
        y_test == 0
    ]

    fraud_probs = probabilities[
        y_test == 1
    ]

    plt.figure(
        figsize=(12, 7)
    )

    sns.kdeplot(
        legit_probs,
        fill=True,
        label="Legitimate Transactions",
        alpha=0.6
    )

    sns.kdeplot(
        fraud_probs,
        fill=True,
        label="Fraud Transactions",
        alpha=0.6
    )

    plt.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}"
    )

    plt.xlabel(
        "Predicted Fraud Probability"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Fraud vs Legitimate Probability Distribution"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR /
        "probability_distribution.png",
        dpi=300
    )

    plt.show()

    plt.close()

    print(
        "\nProbability Distribution Saved Successfully\n"
    )


if __name__ == "__main__":

    run_probability_distribution()