import json
from datetime import datetime

from src.ingestion.sqlite_loader import (
    load_transactions
)

from src.utils.paths import (
    INGESTION_DIR
)

from src.utils.logger import (
    get_logger
)

logger = get_logger(__name__)


def run_ingestion():

    logger.info(
        "Starting ingestion"
    )

    df = load_transactions()

    csv_path = (
        INGESTION_DIR /
        "transactions.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    metadata = {

        "rows": int(len(df)),

        "columns": int(
            len(df.columns)
        ),

        "created_at":
            datetime.now()
            .isoformat(),

        "source":
            "fraud.db"
    }

    metadata_path = (
        INGESTION_DIR /
        "ingestion_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    logger.info(
        "Ingestion completed"
    )

    logger.info(
        f"Saved: {csv_path}"
    )

    return csv_path