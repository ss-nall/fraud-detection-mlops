import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)

from src.utils.paths import (
    PREPROCESSING_DIR,
    THRESHOLD_DIR,
    MODELS_DIR
)


def run_threshold_tuning():

    mlflow.set_experiment(
        "Fraud Detection"
    )

    with mlflow.start_run(
        run_name="threshold_tuning"
    ):

        model = joblib.load(
            MODELS_DIR /
            "random_search_model.pkl"
        )

        X_test = pd.read_csv(
            PREPROCESSING_DIR /
            "X_test.csv"
        )

        y_test = pd.read_csv(
            PREPROCESSING_DIR /
            "y_test.csv"
        ).squeeze()

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        thresholds = np.arange(
            0.01,
            0.99,
            0.01
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        pr_auc = average_precision_score(
            y_test,
            y_prob
        )

        results = []

        for threshold in thresholds:

            y_pred = (
                y_prob >= threshold
            ).astype(int)

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            fbeta = fbeta_score(
                y_test,
                y_pred,
                beta=1.5,
                zero_division=0
            )

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                y_pred
            ).ravel()

            results.append({

                "threshold":
                    threshold,

                "accuracy":
                    accuracy,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "fbeta":
                    fbeta,

                "roc_auc":
                    roc_auc,

                "pr_auc":
                    pr_auc,

                "tp":
                    int(tp),

                "tn":
                    int(tn),

                "fp":
                    int(fp),

                "fn":
                    int(fn)

            })

        threshold_df = pd.DataFrame(
            results
        )

        threshold_df.to_csv(
            THRESHOLD_DIR /
            "threshold_results.csv",
            index=False
        )

        candidate_df = threshold_df[

            (
                threshold_df["precision"]
                >= 0.85
            )

            &

            (
                threshold_df["recall"]
                >= 0.75
            )

        ]

        if len(candidate_df) > 0:

            best_row = candidate_df.loc[

                candidate_df[
                    "fbeta"
                ].idxmax()

            ]

        else:

            best_row = threshold_df.loc[

                threshold_df[
                    "fbeta"
                ].idxmax()

            ]

        best_threshold = float(
            best_row[
                "threshold"
            ]
        )

        

        joblib.dump(

            best_threshold,

            MODELS_DIR /
            "optimal_threshold.pkl"

        )



        final_predictions = (

            y_prob >= best_threshold

        ).astype(int)

        cm = confusion_matrix(
            y_test,
            final_predictions
        )

        tn, fp, fn, tp = cm.ravel()

        final_metrics = {

            "threshold":
                best_threshold,

            "accuracy":
                float(
                    accuracy_score(
                        y_test,
                        final_predictions
                    )
                ),

            "precision":
                float(
                    precision_score(
                        y_test,
                        final_predictions
                    )
                ),

            "recall":
                float(
                    recall_score(
                        y_test,
                        final_predictions
                    )
                ),

            "f1":
                float(
                    f1_score(
                        y_test,
                        final_predictions
                    )
                ),

            "roc_auc":
                float(
                    roc_auc_score(
                        y_test,
                        y_prob
                    )
                ),

            "pr_auc":
                float(
                    average_precision_score(
                        y_test,
                        y_prob
                    )
                ),

            "tp":
                int(tp),

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn)

        }

        with open(
            THRESHOLD_DIR /
            "threshold_metrics.json",
            "w"
        ) as f:

            json.dump(
                final_metrics,
                f,
                indent=4
            )

        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.tight_layout()

        plt.savefig(
            THRESHOLD_DIR /
            "threshold_confusion_matrix.png"
        )

        plt.close()

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            threshold_df[
                "threshold"
            ],
            threshold_df[
                "precision"
            ],
            label="Precision"
        )

        plt.plot(
            threshold_df[
                "threshold"
            ],
            threshold_df[
                "recall"
            ],
            label="Recall"
        )

        plt.plot(
            threshold_df[
                "threshold"
            ],
            threshold_df[
                "f1"
            ],
            label="F1 Score"
        )

        plt.axvline(
            best_threshold,
            linestyle="--",
            label=f"Best = {best_threshold:.2f}"
        )

        plt.xlabel(
            "Threshold"
        )

        plt.ylabel(
            "Score"
        )

        plt.title(
            "Threshold Tuning"
        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            THRESHOLD_DIR /
            "threshold_curve.png"
        )

        plt.close()

        print(
            "\nBest Threshold Selected:"
        )

        print(
            best_threshold
        )

        print(
            "\nFinal Metrics:\n"
        )

        print(
            json.dumps(
                final_metrics,
                indent=4
            )
        )

        mlflow.log_metric(
            "optimal_threshold",
            best_threshold
        )

        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        mlflow.log_artifact(
            THRESHOLD_DIR /
            "threshold_curve.png"
        )

        mlflow.log_artifact(
            THRESHOLD_DIR /
            "threshold_confusion_matrix.png"
        )

        return best_threshold


if __name__ == "__main__":

    print(
        "\nRunning Threshold Tuning...\n"
    )

    run_threshold_tuning()