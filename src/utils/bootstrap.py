from src.utils.helpers import create_directories

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
    INFERENCE_DIR,
    MODELS_DIR
)


def initialize_project():

    create_directories([

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

        INFERENCE_DIR,

        MODELS_DIR

    ])