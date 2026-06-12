import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split
)

from src.utils.config import (
    RANDOM_STATE,
    TEST_SIZE
)

from src.utils.paths import (
    FEATURE_DIR,
    PREPROCESSING_DIR,
    MODELS_DIR
)


def run_preprocessing():

    input_path = (
        FEATURE_DIR /
        "featured_transactions.csv"
    )

    df = pd.read_csv(
        input_path
    )

    drop_columns = [

        "transaction_id",

        "timestamp",

        "sender_id",

        "receiver_id",

        "device_id"
    ]

    df = df.drop(
        columns=drop_columns
    )

    df = pd.get_dummies(
        df,
        columns=[
            "transaction_type"
        ],
        drop_first=False
    )

    df = df.fillna(0)

    y = df["is_fraud"]

    X = df.drop(
        columns=["is_fraud"]
    )

    feature_names = list(
        X.columns
    )

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    smote = SMOTE(
        random_state=RANDOM_STATE
    )

    X_train_smote, y_train_smote = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    X_train.to_csv(
        PREPROCESSING_DIR /
        "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PREPROCESSING_DIR /
        "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PREPROCESSING_DIR /
        "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PREPROCESSING_DIR /
        "y_test.csv",
        index=False
    )

    X_train_smote.to_csv(
        PREPROCESSING_DIR /
        "X_train_smote.csv",
        index=False
    )

    y_train_smote.to_csv(
        PREPROCESSING_DIR /
        "y_train_smote.csv",
        index=False
    )

    joblib.dump(

        feature_names,

        MODELS_DIR /
        "model_features.pkl"
    )

    print(
        "Preprocessing Completed"
    )