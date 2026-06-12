import sqlite3
import pandas as pd

from src.utils.paths import DATABASE_PATH


def load_transactions():

    conn = sqlite3.connect(DATABASE_PATH)

    query = """
    SELECT *
    FROM transactions
    """

    df = pd.read_sql(
        query,
        conn
    )

    conn.close()

    return df