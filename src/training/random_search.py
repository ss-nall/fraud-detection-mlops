import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from lightgbm import LGBMClassifier

from sklearn.model_selection import RandomizedSearchCV

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
    TUNING_DIR,
    MODELS_DIR
)

from src.utils.config import (
    RANDOM_STATE
)


def run_random_search():

    mlflow.set_experiment(
        "Fraud Detection"
    )

    with mlflow.start_run(
        run_name="random_search"
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
            random_state=RANDOM_STATE
        )

        param_grid = {

            "n_estimators":
                [100, 200, 300, 500],

            "learning_rate":
                [0.01, 0.03, 0.05, 0.1],

            "num_leaves":
                [15, 31, 63, 127],

            "max_depth":
                [-1, 5, 10, 15],

            "min_child_samples":
                [10, 20, 50, 100],

            "subsample":
                [0.7, 0.8, 0.9, 1.0],

            "colsample_bytree":
                [0.7, 0.8, 0.9, 1.0]
        }

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=param_grid,

            n_iter=50,

            scoring="roc_auc",

            cv=3,

            verbose=2,

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

        search.fit(
            X_train,
            y_train
        )

        best_model = search.best_estimator_

        with open(
            TUNING_DIR /
            "best_random_search_params.json",
            "w"
        ) as f:

            json.dump(
            search.best_params_,
            f,
            indent=4
            )

        y_pred = best_model.predict(
            X_test
        )

        y_prob = best_model.predict_proba(
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

        with open(
            TUNING_DIR /
            "random_search_metrics.json",
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        pd.DataFrame(
            search.cv_results_
        ).to_csv(

            TUNING_DIR /
            "random_search_results.csv",

            index=False
        )

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
            TUNING_DIR /
            "random_search_confusion_matrix.png"
        )

        plt.close()

        joblib.dump(

            best_model,

            MODELS_DIR /
            "random_search_model.pkl"
        )

        print(
            "\nBest Parameters:\n"
        )

        print(
            search.best_params_
        )

        mlflow.log_params(
            search.best_params_
        )

        print(
            "\nMetrics:\n"
        )

        print(
            json.dumps(
                metrics,
                indent=4
            )
        )

        for metric_name, value in metrics.items():

            mlflow.log_metric(
                metric_name,
                value
            )
        
        mlflow.log_artifact(
            TUNING_DIR /
            "random_search_confusion_matrix.png"
        )

        mlflow.log_artifact(
            TUNING_DIR /
            "random_search_results.csv"
        )

        return best_model
    
if __name__ == "__main__":

    print(
        "\nRunning Random Search...\n"
    )

    run_random_search()