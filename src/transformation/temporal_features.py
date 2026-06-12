import pandas as pd


def create_temporal_features(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["is_night"] = (
        (
            df["hour"] >= 22
        )
        |
        (
            df["hour"] <= 5
        )
    ).astype(int)

    df["is_high_risk_hour"] = (
        (
            df["hour"] >= 0
        )
        &
        (
            df["hour"] <= 4
        )
    ).astype(int)

    df["is_weekend_night"] = (
        (
            df["is_weekend"] == 1
        )
        &
        (
            df["is_night"] == 1
        )
    ).astype(int)

    df["is_high_amount_at_night"] = (
        (
            df["amount"] > 100000
        )
        &
        (
            df["is_night"] == 1
        )
    ).astype(int)

    df["is_high_amount_at_midnight"] = (
        (
            df["amount"] > 100000
        )
        &
        (
            df["hour"] <= 2
        )
    ).astype(int)

    return df