import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.utils.paths import (
    PREPROCESSING_DIR,
    SHAP_DIR,
    MODELS_DIR
)


def run_shap_analysis():

    model = joblib.load(
        MODELS_DIR /
        "random_search_model.pkl"
    )

    X_test = pd.read_csv(
        PREPROCESSING_DIR /
        "X_test.csv"
    )

    sample_size = min(
        2000,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    if isinstance(
        shap_values,
        list
    ):
        shap_values = shap_values[1]

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_sample,

        show=False

    )

    plt.tight_layout()

    plt.savefig(
        SHAP_DIR /
        "shap_summary.png",
        bbox_inches="tight"
    )

    plt.close()

    plt.figure()

    shap.summary_plot(

        shap_values,

        X_sample,

        plot_type="bar",

        show=False

    )

    plt.tight_layout()

    plt.savefig(
        SHAP_DIR /
        "shap_bar.png",
        bbox_inches="tight"
    )

    plt.close()

    feature_importance = pd.DataFrame({

        "feature":
            X_sample.columns,

        "mean_abs_shap":
            abs(
                shap_values
            ).mean(
                axis=0
            )

    })

    feature_importance = (

        feature_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )

    )

    feature_importance.to_csv(

        SHAP_DIR /
        "feature_importance.csv",

        index=False
    )

    print(
        "\nTop SHAP Features:\n"
    )

    print(
        feature_importance.head(20)
    )


if __name__ == "__main__":

    run_shap_analysis()