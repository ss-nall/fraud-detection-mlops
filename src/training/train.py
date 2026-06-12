import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.utils.paths import (
    PREPROCESSING_DIR,
    TRAINING_DIR,
    MODELS_DIR
)

from src.utils.config import (
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE,
    NUM_LEAVES,
    MAX_DEPTH
)


def run_training():

    mlflow.set_experiment(
        "Fraud Detection"
    )

    with mlflow.start_run(
        run_name="baseline_model"
    ):

        X_train = pd.read_csv(
            PREPROCESSING_DIR /
            "X_train_smote.csv"
        )

        y_train = pd.read_csv(
            PREPROCESSING_DIR /
            "y_train_smote.csv"
        ).squeeze()

        X_test = pd.read_csv(
            PREPROCESSING_DIR /
            "X_test.csv"
        )

        y_test = pd.read_csv(
            PREPROCESSING_DIR /
            "y_test.csv"
        ).squeeze()

        model = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            max_depth=MAX_DEPTH
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        metrics = {

            "accuracy":
                float(
                    accuracy_score(
                        y_test,
                        y_pred
                    )
                ),

            "precision":
                float(
                    precision_score(
                        y_test,
                        y_pred
                    )
                ),

            "recall":
                float(
                    recall_score(
                        y_test,
                        y_pred
                    )
                ),

            "f1":
                float(
                    f1_score(
                        y_test,
                        y_pred
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
                )
        }

        for metric_name, value in metrics.items():

            mlflow.log_metric(
                metric_name,
                value
            )

        with open(
            TRAINING_DIR /
            "baseline_metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        joblib.dump(
            model,
            MODELS_DIR /
            "baseline_model.pkl"
        )

        mlflow.log_artifact(
            TRAINING_DIR /
            "baseline_metrics.json"
        )

        mlflow.log_artifact(
            TRAINING_DIR /
            "baseline_confusion_matrix.png"
        )

        importance_df = pd.DataFrame({

            "feature":
                X_train.columns,

            "importance":
                model.feature_importances_

        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
        )

        importance_df.to_csv(
            TRAINING_DIR /
            "feature_importance.csv",
            index=False
        )

        plt.figure(
            figsize=(10, 8)
        )

        top_features = (
            importance_df
            .head(20)
        )

        plt.barh(
            top_features["feature"],
            top_features["importance"]
        )

        plt.tight_layout()

        plt.savefig(
            TRAINING_DIR /
            "feature_importance.png"
        )

        plt.close()

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.tight_layout()

        plt.savefig(
            TRAINING_DIR /
            "baseline_confusion_matrix.png"
        )

        plt.close()

        print(
            json.dumps(
                metrics,
                indent=4
            )
        )

        return model

if __name__ == "__main__":

    print(
        "\nRunning Training...\n"
    )

    run_training()