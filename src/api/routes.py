import json
import tempfile

from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import Response

from prometheus_client import (
    Counter,
    generate_latest
)

from src.serving.inference import (
    run_inference
)

from src.api.schemas import (
    HealthResponse,
    PredictionResponse
)


router = APIRouter()


prediction_counter = Counter(
    "fraud_predictions_total",
    "Total fraud prediction requests"
)

health_counter = Counter(
    "health_checks_total",
    "Total health check requests"
)


@router.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    health_counter.inc()

    return {
        "status": "healthy"
    }


@router.post(
    "/predict-file",
    response_model=PredictionResponse
)
async def predict_file(

    file: UploadFile = File(...)

):

    prediction_counter.inc()

    if not (

        file.filename.endswith(".csv")

        or

        file.filename.endswith(".json")

    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Only CSV and JSON files are supported."
            )

        )

    suffix = Path(
        file.filename
    ).suffix

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=suffix

    ) as temp_file:

        contents = await file.read()

        temp_file.write(
            contents
        )

        temp_path = temp_file.name

    try:

        output = run_inference(
            temp_path
        )

        summary = output[
            "summary"
        ]

        return {

            "message":
                "Inference completed successfully",

            "output_file":
                "artifacts/inference/inference_predictions.csv",

            **summary

        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


@router.get("/metrics")
def metrics():

    return Response(

        content=generate_latest(),

        media_type="text/plain"

    )