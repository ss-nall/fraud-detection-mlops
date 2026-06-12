import numpy as np


def create_amount_features(df):

    df["log_amount"] = np.log1p(
        df["amount"]
    )

    df[
        "user_avg_amount"
    ] = (

        df.groupby("sender_id")[
            "amount"
        ]
        .transform("mean")

    )

    df[
        "user_amount_std"
    ] = (

        df.groupby("sender_id")[
            "amount"
        ]
        .transform("std")

    )

    df[
        "user_amount_std"
    ] = df[
        "user_amount_std"
    ].fillna(0)

    df[
        "relative_amount"
    ] = (

        df["amount"]

        /

        (
            df[
                "user_avg_amount"
            ]
            + 1
        )

    )

    df[
        "amount_to_balance_ratio"
    ] = (

        df["amount"]

        /

        (
            df[
                "sender_balance_before"
            ]
            + 1
        )

    )

    df["drain_rate"] = (

        (
            df[
                "sender_balance_before"
            ]

            -

            df[
                "sender_balance_after"
            ]

        )

        /

        (
            df[
                "sender_balance_before"
            ]
            + 1
        )

    )

    df["is_amount_spike"] = (
        df["relative_amount"] > 3
    ).astype(int)

    df[
        "is_high_value_transaction"
    ] = (
        df["amount"] > 100000
    ).astype(int)

    return df