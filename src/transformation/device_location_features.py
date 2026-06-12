import numpy as np


def create_device_location_features(df):

    df[
        "device_shared_users"
    ] = (

        df.groupby("device_id")[
            "sender_id"
        ]
        .transform("nunique")

    )

    df["device_usage_count"] = (

        df.groupby("device_id")[
            "transaction_id"
        ]
        .transform("count")

    )

    df["new_device_flag"] = (
        df[
            "device_usage_count"
        ] <= 1
    ).astype(int)

    df[
        "is_new_device_high_amount"
    ] = (
        (
            df["new_device_flag"]
            == 1
        )
        &
        (
            df["amount"] > 100000
        )
    ).astype(int)

    df["is_out_of_kenya"] = (
        (
            df["location_lat"] < -4.7
        )
        |
        (
            df["location_lat"] > 5.5
        )
        |
        (
            df["location_lon"] < 33.5
        )
        |
        (
            df["location_lon"] > 42.0
        )
    ).astype(int)


    df["geo_variance"] = (

        df.groupby("sender_id")[
            "location_lat"
        ]
        .transform("std")
        .fillna(0)

    )

    return df