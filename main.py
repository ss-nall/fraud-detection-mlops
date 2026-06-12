from pathlib import Path

from src.utils.bootstrap import (
    initialize_project
)

from src.ingestion.ingest import (
    run_ingestion
)

from src.validation.validate import (
    run_validation
)

from src.transformation.feature_pipeline import (
    run_feature_engineering
)

from src.transformation.preprocessing import (
    run_preprocessing
)

from src.training.train import (
    run_training
)

from src.training.random_search import (
    run_random_search
)

from src.training.threshold_tuning import (
    run_threshold_tuning
)

from src.evaluation.evaluate import (
    run_evaluation
)

from src.evaluation.shap_analysis import (
    run_shap_analysis
)

from src.evaluation.model_registry import (
    run_model_registry
)

from src.prediction.test_prediction import (
    run_prediction_test
)

from src.utils.paths import (
    INGESTION_DIR,
    VALIDATION_DIR,
    FEATURE_DIR,
    PREPROCESSING_DIR,
    TRAINING_DIR,
    TUNING_DIR,
    THRESHOLD_DIR,
    EVALUATION_DIR,
    SHAP_DIR,
    MODEL_REGISTRY_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR
)


def verify_file(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing artifact: {file_path}"
        )

    print(
        f"[OK] {file_path}"
    )


def main():

    print(
        "\nInitializing Project...\n"
    )

    initialize_project()

    print(
        "\nRunning Ingestion...\n"
    )

    run_ingestion()

    verify_file(
        INGESTION_DIR /
        "transactions.csv"
    )

    verify_file(
        INGESTION_DIR /
        "ingestion_metadata.json"
    )

    print(
        "\nRunning Validation...\n"
    )

    run_validation()

    verify_file(
        VALIDATION_DIR /
        "validated_transactions.csv"
    )

    verify_file(
        VALIDATION_DIR /
        "validation_report.json"
    )

    print(
        "\nRunning Feature Engineering...\n"
    )

    run_feature_engineering()

    verify_file(
        FEATURE_DIR /
        "featured_transactions.csv"
    )

    verify_file(
        FEATURE_DIR /
        "feature_metadata.json"
    )

    print(
        "\nRunning Preprocessing...\n"
    )

    run_preprocessing()

    verify_file(
        PREPROCESSING_DIR /
        "X_train.csv"
    )

    verify_file(
        PREPROCESSING_DIR /
        "X_test.csv"
    )

    verify_file(
        PREPROCESSING_DIR /
        "y_train.csv"
    )

    verify_file(
        PREPROCESSING_DIR /
        "y_test.csv"
    )

    verify_file(
        PREPROCESSING_DIR /
        "X_train_smote.csv"
    )

    verify_file(
        PREPROCESSING_DIR /
        "y_train_smote.csv"
    )

    verify_file(
        MODELS_DIR /
        "model_features.pkl"
    )

    print(
        "\nRunning Baseline Training...\n"
    )

    run_training()

    verify_file(
        TRAINING_DIR /
        "baseline_metrics.json"
    )

    verify_file(
        TRAINING_DIR /
        "baseline_confusion_matrix.png"
    )

    verify_file(
        TRAINING_DIR /
        "feature_importance.csv"
    )

    verify_file(
        TRAINING_DIR /
        "feature_importance.png"
    )

    verify_file(
        MODELS_DIR /
        "baseline_model.pkl"
    )

    print(
        "\nRunning Random Search...\n"
    )

    run_random_search()

    verify_file(
        TUNING_DIR /
        "random_search_metrics.json"
    )

    verify_file(
        TUNING_DIR /
        "random_search_results.csv"
    )

    verify_file(
        TUNING_DIR /
        "random_search_confusion_matrix.png"
    )

    verify_file(
        MODELS_DIR /
        "random_search_model.pkl"
    )

    print(
        "\nRunning Threshold Tuning...\n"
    )

    run_threshold_tuning()

    verify_file(
        THRESHOLD_DIR /
        "threshold_results.csv"
    )

    verify_file(
        THRESHOLD_DIR /
        "threshold_metrics.json"
    )

    verify_file(
        THRESHOLD_DIR /
        "threshold_confusion_matrix.png"
    )

    verify_file(
        THRESHOLD_DIR /
        "threshold_curve.png"
    )

    verify_file(
        MODELS_DIR /
        "optimal_threshold.pkl"
    )

    print(
        "\nRunning Evaluation...\n"
    )

    run_evaluation()

    verify_file(
        EVALUATION_DIR /
        "metrics.json"
    )

    verify_file(
        EVALUATION_DIR /
        "classification_report.txt"
    )

    verify_file(
        EVALUATION_DIR /
        "confusion_matrix.png"
    )

    verify_file(
        EVALUATION_DIR /
        "roc_curve.png"
    )

    verify_file(
        EVALUATION_DIR /
        "pr_curve.png"
    )

    verify_file(
        EVALUATION_DIR /
        "ks_curve.png"
    )

    print(
        "\nRunning SHAP Analysis...\n"
    )

    run_shap_analysis()

    verify_file(
        SHAP_DIR /
        "shap_summary.png"
    )

    verify_file(
        SHAP_DIR /
        "shap_bar.png"
    )

    verify_file(
        SHAP_DIR /
        "feature_importance.csv"
    )

    print(
        "\nRunning Model Registry...\n"
    )

    run_model_registry()

    verify_file(
        MODEL_REGISTRY_DIR /
        "model_card.json"
    )

    print(
        "\nRunning Prediction Validation...\n"
    )

    run_prediction_test()

    verify_file(
        PREDICTIONS_DIR /
        "predictions.csv"
    )

    verify_file(
        PREDICTIONS_DIR /
        "prediction_metrics.json"
    )

    verify_file(
        PREDICTIONS_DIR /
        "prediction_classification_report.txt"
    )

    verify_file(
        PREDICTIONS_DIR /
        "prediction_confusion_matrix.png"
    )

    print(
        "\nPipeline Completed Successfully\n"
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            f"\nPipeline Failed: {e}"
        )

        raise