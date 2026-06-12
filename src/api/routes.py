import json
import tempfile

from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from src.serving.inference import (
    run_inference
)

from src.api.schemas import (
    HealthResponse,
    PredictionResponse
)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

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