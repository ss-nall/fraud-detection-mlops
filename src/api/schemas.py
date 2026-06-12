from pydantic import BaseModel


class HealthResponse(
    BaseModel
):

    status: str


class PredictionResponse(
    BaseModel
):

    message: str

    output_file: str

    total_transactions: int

    fraud_predictions: int

    legitimate_predictions: int

    fraud_percentage: float

    average_fraud_probability: float

    threshold: float