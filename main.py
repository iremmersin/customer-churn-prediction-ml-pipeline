import logging
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional
import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from churn_model import ChurnPredictor

logger = logging.getLogger("churn_api")

MODEL_PATH = os.getenv("MODEL_PATH", "churn_model.pkl")

N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook/customer-churn"
)

N8N_TIMEOUT = float(os.getenv("N8N_TIMEOUT", "10"))

predictor = ChurnPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):

    predictor.load_model(MODEL_PATH)
    logger.info("API is ready and model is loaded.")

    yield


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "AI-powered customer churn prediction API "
        "with SHAP explanations and business recommendations."
    ),
    version="4.0",
    lifespan=lifespan
)

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]

class CustomerData(BaseModel):

    customer_id: Optional[str] = Field(
        default=None,
        description="Optional identifier, echoed back so downstream workflows can act on the result."
    )

    tenure: int = Field(
        ge=0,
        description="Number of months the customer has stayed."
    )

    MonthlyCharges: float = Field(
        ge=0,
        description="Monthly customer charge."
    )

    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total amount charged. May be omitted for brand-new customers."
    )

    SeniorCitizen: int = Field(
        ge=0,
        le=1
    )

    Contract: Literal["Month-to-month", "One year", "Two year"]

    InternetService: Literal["DSL", "Fiber optic", "No"]

    TechSupport: YesNoNoInternet

    OnlineSecurity: YesNoNoInternet

    OnlineBackup: YesNoNoInternet

    DeviceProtection: YesNoNoInternet

    StreamingTV: YesNoNoInternet

    StreamingMovies: YesNoNoInternet

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    PaperlessBilling: YesNo

    Partner: YesNo

    Dependents: YesNo

    MultipleLines: Literal["Yes", "No", "No phone service"]

    PhoneService: YesNo

    customer_value: Literal["Low", "Medium", "High"] = Field(
        default="Medium",
        description=(
            "Business-defined customer value. "
            "Not used by the ML model; only 'High' is eligible for a discount."
        )
    )

def send_to_n8n(data):
    
    try:

        response = requests.post(
            N8N_WEBHOOK_URL,
            json=data,
            timeout=N8N_TIMEOUT
        )

        response.raise_for_status()

        logger.info("Prediction result sent to n8n successfully.")

        return True

    except requests.RequestException as e:

        logger.warning("Failed to send prediction to n8n: %s", e)

        return False

@app.get("/")
def root():

    return {
        "message": "Telco Customer Churn Prediction API",
        "status": "running",
        "model": MODEL_PATH
    }

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": (
            predictor.model is not None
            and predictor.preprocessor is not None
        )
    }

@app.post("/predict")
def predict(
    customer: CustomerData,
    background_tasks: BackgroundTasks
):

    try:

        payload = customer.model_dump()

        customer_id = payload.pop("customer_id", None)

        result = predictor.predict_with_explanation(pd.DataFrame([payload]))

        result["customer_id"] = customer_id

        # Runs after the response is sent, so a slow n8n never delays the caller.
        background_tasks.add_task(send_to_n8n, result)

        return result

    except Exception as e:

        logger.exception("Prediction failed")

        raise HTTPException(status_code=500, detail=str(e))