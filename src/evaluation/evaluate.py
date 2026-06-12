import json
import joblib
import numpy as np
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
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from src.utils.paths import (
    PREPROCESSING_DIR,
    EVALUATION_DIR,
    MODELS_DIR
)


def run_evaluation():

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

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

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
            float(pr_auc)

    }

    with open(
        EVALUATION_DIR /
        "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    report = classification_report(
        y_test,
        y_pred
    )

    with open(
        EVALUATION_DIR /
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

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
        EVALUATION_DIR /
        "confusion_matrix.png"
    )

    plt.close()

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR /
        "roc_curve.png"
    )

    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure()

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"PR AUC = {pr_auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR /
        "pr_curve.png"
    )

    plt.close()

    fraud_scores = y_prob[
        y_test == 1
    ]

    nonfraud_scores = y_prob[
        y_test == 0
    ]

    thresholds = np.sort(
        np.unique(y_prob)
    )

    ks_values = []

    for threshold in thresholds:

        fraud_cdf = np.mean(
            fraud_scores <= threshold
        )

        nonfraud_cdf = np.mean(
            nonfraud_scores <= threshold
        )

        ks_values.append(
            abs(
                fraud_cdf -
                nonfraud_cdf
            )
        )

    ks_statistic = float(
        max(ks_values)
    )

    plt.figure()

    plt.plot(
        thresholds,
        ks_values
    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        "KS Statistic"
    )

    plt.title(
        f"KS Curve (Max KS = {ks_statistic:.4f})"
    )

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR /
        "ks_curve.png"
    )

    plt.close()

    metrics["ks_statistic"] = ks_statistic

    with open(
        EVALUATION_DIR /
        "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )


if __name__ == "__main__":

    run_evaluation()