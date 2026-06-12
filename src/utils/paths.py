from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================
# DATA
# ============================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"

# ============================================
# ARTIFACTS
# ============================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

INGESTION_DIR = ARTIFACTS_DIR / "ingestion"

VALIDATION_DIR = ARTIFACTS_DIR / "validation"

FEATURE_DIR = ARTIFACTS_DIR / "features"

PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"

TRAINING_DIR = ARTIFACTS_DIR / "training"

TUNING_DIR = ARTIFACTS_DIR / "tuning"

THRESHOLD_DIR = ARTIFACTS_DIR / "threshold"

EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

SHAP_DIR = ARTIFACTS_DIR / "shap"

MODEL_REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"

PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# ============================================
# MODELS
# ============================================

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================
# DATABASE
# ============================================

DATABASE_PATH = RAW_DATA_DIR / "fraud.db"

INFERENCE_DIR = (
    ARTIFACTS_DIR /
    "inference"
)