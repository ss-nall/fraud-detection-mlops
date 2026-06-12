import json

import pandas as pd

from src.validation.schema import (
    COLUMN_SCHEMA
)

from src.utils.paths import (
    INGESTION_DIR,
    VALIDATION_DIR
)

from src.utils.logger import (
    get_logger
)

logger = get_logger(__name__)


def run_validation():

    logger.info(
        "Starting validation"
    )

    input_path = (
        INGESTION_DIR /
        "transactions.csv"
    )

    df = pd.read_csv(
        input_path
    )

    report = {}

    required_columns = list(
        COLUMN_SCHEMA.keys()
    )

    missing_columns = [

        col

        for col in required_columns

        if col not in df.columns
    ]

    report["missing_columns"] = (
        missing_columns
    )

    report["row_count"] = (
        int(len(df))
    )

    report["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    report["missing_values"] = {

        col: int(val)

        for col, val in
        df.isnull().sum().items()
    }

    dtype_issues = {}

    for column, rules in COLUMN_SCHEMA.items():

        if column not in df.columns:
            continue

        actual_dtype = str(
            df[column].dtype
        )

        expected_dtype = rules["dtype"]

        if isinstance(
            expected_dtype,
            str
        ):
            expected_dtype = [
                expected_dtype
            ]

        if actual_dtype not in expected_dtype:

            dtype_issues[column] = {

                "expected":
                    expected_dtype,

                "actual":
                    actual_dtype
            }

    report["dtype_issues"] = (
        dtype_issues
    )

    range_issues = {}

    for column, rules in COLUMN_SCHEMA.items():

        if column not in df.columns:
            continue

        if "min" in rules:

            invalid_min = int(

                (
                    df[column]
                    < rules["min"]
                ).sum()

            )

            if invalid_min > 0:

                range_issues[
                    f"{column}_min"
                ] = invalid_min

        if "max" in rules:

            invalid_max = int(

                (
                    df[column]
                    > rules["max"]
                ).sum()

            )

            if invalid_max > 0:

                range_issues[
                    f"{column}_max"
                ] = invalid_max

    report["range_issues"] = (
        range_issues
    )

    fraud_label_issues = int(

        (
            ~df["is_fraud"]
            .isin([0, 1])
        ).sum()

    )

    report[
        "invalid_fraud_labels"
    ] = fraud_label_issues

    try:

        pd.to_datetime(
            df["timestamp"]
        )

        report[
            "timestamp_valid"
        ] = True

    except Exception:

        report[
            "timestamp_valid"
        ] = False

    df = df.drop_duplicates()

    output_csv = (
        VALIDATION_DIR /
        "validated_transactions.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    output_json = (
        VALIDATION_DIR /
        "validation_report.json"
    )

    with open(
        output_json,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    logger.info(
        "Validation completed"
    )

    logger.info(
        f"Saved: {output_csv}"
    )

    logger.info(
        f"Saved: {output_json}"
    )

    return output_csv