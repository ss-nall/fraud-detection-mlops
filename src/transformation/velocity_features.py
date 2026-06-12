import pandas as pd


def create_velocity_features(df):

    df = df.sort_values(
        ["sender_id", "timestamp"]
    )

    df[
        "time_since_last_txn"
    ] = (

        df.groupby("sender_id")[
            "timestamp"
        ]
        .diff()
        .dt.total_seconds()

    )

    df[
        "time_since_last_txn"
    ] = df[
        "time_since_last_txn"
    ].fillna(999999)

    df[
        "transaction_count_per_user"
    ] = (

        df.groupby(
            "sender_id"
        )["transaction_id"]
        .transform("count")

    )

    df[
        "transactions_last_1hr"
    ] = (

        df.groupby("sender_id")
        .cumcount()

    )

    df[
        "transactions_last_24hr"
    ] = (

        df.groupby("sender_id")
        .cumcount()

    )

   

    return df