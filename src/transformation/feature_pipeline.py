import json
import pandas as pd

from src.utils.paths import (
    VALIDATION_DIR,
    FEATURE_DIR
)

from src.transformation.temporal_features import (
    create_temporal_features
)

from src.transformation.velocity_features import (
    create_velocity_features
)

from src.transformation.amount_features import (
    create_amount_features
)

from src.transformation.behavior_features import (
    create_behavior_features
)

from src.transformation.device_location_features import (
    create_device_location_features
)


def run_feature_engineering():

    input_path = (
        VALIDATION_DIR /
        "validated_transactions.csv"
    )

    df = pd.read_csv(
        input_path
    )

    df = create_temporal_features(df)

    df = create_velocity_features(df)

    df = create_amount_features(df)

    df = create_behavior_features(df)

    df = create_device_location_features(df)

    output_csv = (
        FEATURE_DIR /
        "featured_transactions.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    metadata = {

        "rows": int(len(df)),

        "columns": int(
            len(df.columns)
        )
    }

    with open(
        FEATURE_DIR /
        "feature_metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return output_csv