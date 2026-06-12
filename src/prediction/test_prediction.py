import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.utils.paths import (
    PREPROCESSING_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR
)


def run_prediction_test():

    model = joblib.load(
        MODELS_DIR /
        "random_search_model.pkl"
    )

    threshold = joblib.load(
        MODELS_DIR /
        "optimal_threshold.pkl"
    )

    X_test = pd.read_csv(
        PREPROCESSING_DIR /
        "X_test.csv"
    )

    y_test = pd.read_csv(
        PREPROCESSING_DIR /
        "y_test.csv"
    ).squeeze()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    prediction_df = pd.DataFrame({

        "fraud_probability":
            probabilities,

        "predicted_fraud":
            predictions

    })

    prediction_df.to_csv(

        PREDICTIONS_DIR /
        "predictions.csv",

        index=False

    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {

        "threshold":
            float(threshold),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        "true_positive":
            int(tp),

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "correct_predictions":
            int(tp + tn),

        "incorrect_predictions":
            int(fp + fn),

        "total_transactions":
            int(len(y_test))
    }

    with open(
        PREDICTIONS_DIR /
        "prediction_metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    report = classification_report(
        y_test,
        predictions
    )

    with open(
        PREDICTIONS_DIR /
        "prediction_classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.tight_layout()

    plt.savefig(
        PREDICTIONS_DIR /
        "prediction_confusion_matrix.png"
    )

    plt.close()

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )


if __name__ == "__main__":

    run_prediction_test()