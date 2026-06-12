def create_behavior_features(df):

    df[
        "unique_receivers_count"
    ] = (

        df.groupby("sender_id")[
            "receiver_id"
        ]
        .transform("nunique")

    )

    df[
        "unique_devices_per_user"
    ] = (

        df.groupby("sender_id")[
            "device_id"
        ]
        .transform("nunique")

    )

    df[
        "receiver_diversity_ratio"
    ] = (

        df[
            "unique_receivers_count"
        ]

        /

        (
            df[
                "transaction_count_per_user"
            ]
            + 1
        )

    )

    df[
        "night_transaction_ratio"
    ] = (

        df.groupby("sender_id")[
            "is_night"
        ]
        .transform("mean")

    )

    return df